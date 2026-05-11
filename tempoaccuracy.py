import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import timedelta, datetime
import warnings

warnings.filterwarnings('ignore')  # Potlačit varování

CSV_FILE = 'Germany_metars.csv'  # Cesta k CSV souboru
NUM_DAYS = 0  # Počet dnů na analýzu pro každou stanici
MIN_RECORDS_PER_STATION = 50  # Minimální počet záznamů pro stanici, jinak přeskočit
START_DATE = None  # Filtr: Začátek data (např. '2020-01-01'), None = bez filtru
END_DATE = None  # Filtr: Konec data (např. '2024-12-31'), None = bez filtru
RANDOM_DAYS = False  # True = náhodné dny místo prvních/posledních (vyžaduje NUM_DAYS > 0)
LIMIT_STATIONS = None  # Omezení na počet stanic (např. 10 pro test), None = všechny
GROUP_LENGTH_HOURS = 6  # Maximální délka skupiny v hodinách pro clustering mód (L3 Dopl.5)
MAX_GROUPS_PER_DAY = 4  # Maximální počet skupin na 24h (None = použít clustering podle gapů); pro experimenty měňte (např. 2-6)

def load_data(csv_file):
    """Načte CSV a bezpečně zpracuje sloupce relevantní pro analýzu.
    Očekává sloupce: station, valid, sknt, vsby, gust, wxcodes
    - valid musí být parse_dates
    - vsby se předpokládá v statute miles (SM) nebo v metrech - pokud > 1000, nechá se
    """
    df = pd.read_csv(csv_file, parse_dates=['valid'])

    # Základní převody na numeric (M nebo jiné texty -> NaN)
    df['sknt'] = pd.to_numeric(df.get('sknt'), errors='coerce')
    df['vsby'] = pd.to_numeric(df.get('vsby'), errors='coerce')
    df['gust'] = pd.to_numeric(df.get('gust'), errors='coerce')

    # Převod vsby na metry: pokud hodnota < 1000 předpokládejme, že jde o SM (statute miles)
    # pokud už je vsby v metrech (>1000), necháme ji beze změny.
    def sm_to_m(x):
        if pd.isna(x):
            return np.nan
        try:
            v = float(x)
        except Exception:
            return np.nan
        if v <= 1000:  # pravděpodobně SM -> na metry
            return v * 1609.34
        else:
            return v

    df['vsby_m'] = df['vsby'].apply(sm_to_m)

    # Zaokrouhlení podle L3 (Dopl. 3 Dodatek C):
    #    do 800 m krok 50 m, 800-5000m krok 100 m, >5000 m krok 1000 m (volitelně)
    mask_notna = df['vsby_m'].notna()
    df.loc[mask_notna, 'vsby_m'] = df.loc[mask_notna, 'vsby_m'].astype(float)

    mask_low = df['vsby_m'] <= 800
    mask_mid = (df['vsby_m'] > 800) & (df['vsby_m'] <= 5000)
    mask_high = df['vsby_m'] > 5000

    df.loc[mask_low, 'vsby_m'] = np.round(df.loc[mask_low, 'vsby_m'] / 50) * 50
    df.loc[mask_mid, 'vsby_m'] = np.round(df.loc[mask_mid, 'vsby_m'] / 100) * 100
    df.loc[mask_high, 'vsby_m'] = np.round(df.loc[mask_high, 'vsby_m'] / 1000) * 1000

    # Aplikovat filtr na datum, pokud je zadán
    if START_DATE:
        df = df[df['valid'] >= pd.to_datetime(START_DATE)]
    if END_DATE:
        df = df[df['valid'] <= pd.to_datetime(END_DATE)]

    print(f"Načteno {len(df)} řádků po filtru na datum.")
    print(f"NaN v sknt: {df['sknt'].isna().sum()}, v vsby: {df['vsby_m'].isna().sum()}")

    return df


def categorize_change_wind(base_sknt, base_gust, val_sknt, val_gust):
    """Kategorizace změny větru (rychlost + nárazy, bez směru): 0=bez, 1=significant

    Pravidla (L3 Dopl.5):
      - změna rychlosti ≥ 10 kt -> significant
      - změna nárazu ≥ 10 kt pokud průměr před/po ≥ 15 kt -> significant
    """
    if pd.isna(val_sknt) or pd.isna(base_sknt):
        return 0

    try:
        diff_sknt = abs(float(val_sknt) - float(base_sknt))
    except Exception:
        diff_sknt = 0

    sknt_cat = 1 if diff_sknt >= 10 else 0

    nary_cat = 0
    try:
        avg_sknt = (float(base_sknt) + float(val_sknt)) / 2
    except Exception:
        avg_sknt = 0

    if avg_sknt >= 15:
        val_g = float(val_gust) if not pd.isna(val_gust) else 0
        base_g = float(base_gust) if not pd.isna(base_gust) else 0
        diff_gust = abs(val_g - base_g)
        nary_cat = 1 if diff_gust >= 10 else 0

    return 1 if (sknt_cat or nary_cat) else 0


def categorize_change_vsby(base, val):
    """Kategorizace viditelnosti: 0=bez, 1=significant.
    Používá prahy z L3/Doc8896: 150, 350, 600, 800, 1500, 3000, 5000 (metry)
    """
    if pd.isna(val) or pd.isna(base):
        return 0
    base_f, val_f = float(base), float(val)

    thresholds = [150, 350, 600, 800, 1500, 3000, 5000]
    min_val, max_val = min(base_f, val_f), max(base_f, val_f)

    # Significant pokud byla překročena libovolná mez mezi min a max
    for th in thresholds:
        if th > min_val and th <= max_val:
            return 1
    return 0


def get_wx_cat(code):
    """Kategorizace WX podle L3/Doc8896 (zjednodušeně):
    0 = žádné, 1 = onset / lehké indikátory, 2 = moderate/heavy (významné)
    """
    if pd.isna(code) or code == 'M' or str(code).strip() == '':
        return 0
    code_str = str(code).upper()

    # doplněné termíny (včetně kroup/ledových částic GR/GS)
    precip_terms = ['FZRA', 'FZDZ', 'RA', 'SN', 'DZ', 'TSRA', 'TS', 'SHRA', 'SHSN', 'GR', 'GS']
    storm_terms = ['DS', 'SS']
    onset_terms = ['SQ', 'FC', 'DR', 'DU', 'SA', 'SN']  # onset style events

    # Onset-only terms
    if any(term in code_str for term in onset_terms):
        return 1

    # If obsahuje srážky nebo bouři
    if any(term in code_str for term in precip_terms + storm_terms):
        # pokud je explicitně intenzita + nebo -, bereme to v potaz
        if '+' in code_str:
            return 2
        if '-' in code_str:
            return 1
        # bez prefixu považujme za významné (2)
        return 2

    return 0


def calculate_brier_score(preds, obs):
    """Výpočet Brier score pro binární/probabilistické předpovědi (průměr (p_i - o_i)^2)."""
    if len(preds) == 0:
        return 0.0
    preds = np.array(preds)
    obs = np.array(obs)
    return float(np.mean((preds - obs) ** 2))


def get_reg_prob(fraction):
    """Regulativní pravděpodobnost podle L3 (bez PROB skupin, 100% pro FM/BECMG/TEMPO).
    Předpokládáme, že pokud fraction >= 0.5, použijeme skupinu (FM/BECMG/TEMPO) s p=1.0.
    Pro nižší fraction nemůžeme předpovědět (bez PROB), tak p=0 -> zmeškáme změny.
    """
    return 1.0 if fraction >= 0.5 else 0.0


def analyze_taf_precision(df_station, station, num_days):
    """Hlavní analýza pro jednotlivou stanici.

    - Rozdělí data na dny
    - Pro každý den vezme první záznam jako T0 (výchozí)
    - V následujících ~24 hodinách spočítá ideální detekce (per záznam)
    - Pro regulační: Rozdělí do skupin (fixed bins nebo clustering), pro každou spočítá fraction změn,
      přiřadí p=1.0 všem záznamům v ní pokud fraction >=0.5, jinak 0 (bez PROB)
    - To vede k: snížení detekce pro nízké fraction (omezení L3), FAR >0 (předpověď pro celou skupinu)
    - Vrací metriky (detekce, Brier, confusion matrix, accuracy)
    """
    df_station = df_station.copy()
    df_station['date'] = df_station['valid'].dt.date

    unique_dates = np.array(sorted(df_station['date'].unique()))

    if len(unique_dates) == 0:
        return {'sknt': {'num_changes': 0, 'reg_detected': 0, 'snizeni': 0, 'cm_reg': np.array([[0, 0], [0, 0]]), 'far_reg': 0, 'brier_reg': 0, 'accuracy_reg': 0},
                'vsby': {'num_changes': 0, 'reg_detected': 0, 'snizeni': 0, 'cm_reg': np.array([[0, 0], [0, 0]]), 'far_reg': 0, 'brier_reg': 0, 'accuracy_reg': 0},
                'wx': {'num_changes': 0, 'reg_detected': 0, 'snizeni': 0, 'cm_reg': np.array([[0, 0], [0, 0]]), 'far_reg': 0, 'brier_reg': 0, 'accuracy_reg': 0}}

    # Výběr dnů podle konfigurace
    if RANDOM_DAYS and num_days > 0:
        sel = np.random.choice(unique_dates, min(num_days, len(unique_dates)), replace=False)
        unique_dates = np.sort(sel)
    elif num_days < 0:
        unique_dates = unique_dates[num_days:]  # posledních abs(num_days)
    else:
        unique_dates = unique_dates[:num_days] if num_days > 0 else unique_dates

    # Container pro výsledky
    results_ideal = {'sknt': [], 'vsby': [], 'wx': []}
    results_reg = {'sknt': [], 'vsby': [], 'wx': []}
    real_changes = {'sknt': [], 'vsby': [], 'wx': []}

    for date in unique_dates:
        day_data = df_station[df_station['date'] == date].sort_values('valid').reset_index(drop=True)
        if len(day_data) < 5:
            continue

        base_time = day_data.loc[0, 'valid']
        base_sknt = day_data.loc[0, 'sknt']
        base_gust = day_data.loc[0, 'gust']
        base_vsby = day_data.loc[0, 'vsby_m']
        base_wx_cat = get_wx_cat(day_data.loc[0, 'wxcodes'])

        # Vezmeme následujících 24h od base_time
        future_data = day_data[(day_data['valid'] > base_time) & (day_data['valid'] <= base_time + pd.Timedelta(hours=24))].copy()
        if len(future_data) < 5:
            continue

        future_data = future_data.sort_values('valid').reset_index(drop=True)
        future_data['hours_since_base'] = (future_data['valid'] - base_time).dt.total_seconds() / 3600.0

        # Nejprve spočítat reals a ideal pro všechny záznamy v future_data (ideální per-záznam detekce)
        for _, row in future_data.iterrows():
            val_sknt = row['sknt']
            val_gust = row['gust']
            val_vsby = row['vsby_m']
            val_wx_cat = get_wx_cat(row['wxcodes'])

            r_sknt = categorize_change_wind(base_sknt, base_gust, val_sknt, val_gust) > 0
            r_vsby = categorize_change_vsby(base_vsby, val_vsby) > 0
            r_wx = ((val_wx_cat != 0 and base_wx_cat == 0) or (val_wx_cat > base_wx_cat))

            real_changes['sknt'].append(bool(r_sknt))
            real_changes['vsby'].append(bool(r_vsby))
            real_changes['wx'].append(bool(r_wx))

            results_ideal['sknt'].append(1 if r_sknt else 0)
            results_ideal['vsby'].append(1 if r_vsby else 0)
            results_ideal['wx'].append(1 if r_wx else 0)

        # Nyní vytvořit skupiny pro regulační mód
        future_groups = []
        if MAX_GROUPS_PER_DAY is not None and MAX_GROUPS_PER_DAY > 0:
            # Fixed bins: rovnoměrné rozdělení 24h do MAX_GROUPS_PER_DAY
            num_groups = MAX_GROUPS_PER_DAY
            bin_edges = np.linspace(0, 24, num_groups + 1)
            for i in range(num_groups):
                start_h = bin_edges[i]
                end_h = bin_edges[i + 1]
                mask = (future_data['hours_since_base'] >= start_h) & (future_data['hours_since_base'] < end_h)
                group = future_data[mask].copy()
                if len(group) > 0:
                    future_groups.append(group)
        else:
            # Clustering mód: nová skupina při gap > GROUP_LENGTH_HOURS
            current_group = []
            last_hour = None
            for _, row in future_data.iterrows():
                h = row['hours_since_base']
                if last_hour is None or h - last_hour > GROUP_LENGTH_HOURS:
                    if len(current_group) > 0:
                        future_groups.append(pd.DataFrame(current_group))
                    current_group = [row]
                else:
                    current_group.append(row)
                last_hour = h
            if len(current_group) > 0:
                future_groups.append(pd.DataFrame(current_group))

        # Pro každou skupinu spočítat fractions a přiřadit reg_prob VŠEM záznamům v ní
        for group in future_groups:
            if len(group) == 0:
                continue

            # Počet změn v group pro každý parametr
            sknt_change_count = 0
            vsby_change_count = 0
            wx_change_count = 0

            for _, row in group.iterrows():
                val_sknt = row['sknt']
                val_gust = row['gust']
                val_vsby = row['vsby_m']
                val_wx_cat = get_wx_cat(row['wxcodes'])

                sknt_change_count += categorize_change_wind(base_sknt, base_gust, val_sknt, val_gust) > 0
                vsby_change_count += categorize_change_vsby(base_vsby, val_vsby) > 0
                wx_change_count += ((val_wx_cat != 0 and base_wx_cat == 0) or (val_wx_cat > base_wx_cat))

            group_size = len(group)
            sknt_fraction = sknt_change_count / group_size if group_size > 0 else 0
            vsby_fraction = vsby_change_count / group_size if group_size > 0 else 0
            wx_fraction = wx_change_count / group_size if group_size > 0 else 0

            # Regulativní prob (bez PROB, 100% jen pro >=0.5)
            reg_sknt = get_reg_prob(sknt_fraction)
            reg_vsby = get_reg_prob(vsby_fraction)
            reg_wx = get_reg_prob(wx_fraction)

            # Přiřadit k VŠEM záznamům v group (předpověď pro celou skupinu)
            for _ in range(len(group)):
                results_reg['sknt'].append(reg_sknt)
                results_reg['vsby'].append(reg_vsby)
                results_reg['wx'].append(reg_wx)

    # --- Metriky ---
    metrics = {}
    for param in ['sknt', 'vsby', 'wx']:
        ideal_preds = np.array(results_ideal[param])
        reg_preds = np.array(results_reg[param])
        reals = np.array(real_changes[param]).astype(int)

        num_changes = int(np.sum(reals))
        total_samples = len(reals)
        if num_changes == 0 or total_samples == 0:
            metrics[param] = {'num_changes': 0, 'reg_detected': 0.0, 'snizeni': 0.0, 'cm_reg': np.array([[0, 0], [0, 0]]), 'far_reg': 0.0, 'brier_reg': 0.0, 'accuracy_reg': 0.0}
            continue

        # Detekce: ideální 100%, reg: průměr reg_preds kde reals==1
        ideal_detected = 1.0  # Vždy 100% pro ideální
        reg_detected = float(np.sum(reg_preds[reals == 1]) / num_changes) if num_changes > 0 else 0.0

        snizeni = (1.0 - reg_detected) * 100.0

        # Confusion matrix: threshold na 0.5 (binary pro reg)
        try:
            cm_reg = confusion_matrix(reals, (reg_preds > 0.5).astype(int))
        except Exception:
            cm_reg = np.array([[0, 0], [0, 0]])

        fp_reg = int(cm_reg[0, 1]) if len(cm_reg) == 2 and len(cm_reg[0]) == 2 else 0
        tp_reg = int(cm_reg[1, 1]) if len(cm_reg) == 2 and len(cm_reg[0]) == 2 else 0
        tn_reg = int(cm_reg[0, 0]) if len(cm_reg) == 2 and len(cm_reg[0]) == 2 else 0
        fn_reg = int(cm_reg[1, 0]) if len(cm_reg) == 2 and len(cm_reg[0]) == 2 else 0
        far_reg = float(fp_reg / (fp_reg + tp_reg)) if (fp_reg + tp_reg) > 0 else 0.0

        brier_reg = calculate_brier_score(reg_preds, reals)
        accuracy_reg = float((tp_reg + tn_reg) / total_samples) if total_samples > 0 else 0.0

        metrics[param] = {'num_changes': num_changes, 'reg_detected': reg_detected * 100.0, 'snizeni': snizeni, 'cm_reg': cm_reg, 'far_reg': far_reg, 'brier_reg': brier_reg, 'accuracy_reg': accuracy_reg}

        # Tisk pro parametry
        print(f'\nPro {param.upper()} (stanice {station}):')
        print(f'  Celkový počet změn: {num_changes}')
        print(f'  Předpisová detekce: {metrics[param]["reg_detected"]:.1f}% (ideální 100%)')
        print(f'  Snížení přesnosti kvůli předpisu: {metrics[param]["snizeni"]:.1f}%')
        print(f'  Přesnost (Accuracy) předpisová: {metrics[param]["accuracy_reg"]*100:.1f}%')
        print(f'  Brier score (předpisový): {metrics[param]["brier_reg"]:.3f} (ideální 0.000)')
        print(f'  Confusion Matrix předpisová:\n{cm_reg}')
        print(f'  False Alarm Ratio (FAR): {metrics[param]["far_reg"]:.2f}')

    return metrics


# Hlavní spuštění pro všechny stanice
if __name__ == "__main__":
    df = load_data(CSV_FILE)

    if 'station' not in df.columns:
        raise ValueError("CSV musí obsahovat sloupec 'station'.")

    stations = list(df['station'].unique())
    if LIMIT_STATIONS:
        stations = stations[:LIMIT_STATIONS]

    print(f'Načteno {len(stations)} unikátních stanic (omezeno na {LIMIT_STATIONS} pokud zadáno).')

    summary = []

    for station in stations:
        df_station = df[df['station'] == station].sort_values('valid').reset_index(drop=True)
        print(f'\n=== Analýza pro stanici {station}, počet záznamů: {len(df_station)} ===')

        if len(df_station) < MIN_RECORDS_PER_STATION:
            print(f'Přeskakuji {station} - málo dat (potřeba >= {MIN_RECORDS_PER_STATION}).')
            continue

        metrics = analyze_taf_precision(df_station, station, NUM_DAYS)

        avg_snizeni_sknt = metrics['sknt']['snizeni'] if 'sknt' in metrics else 0.0
        avg_snizeni_vsby = metrics['vsby']['snizeni'] if 'vsby' in metrics else 0.0
        avg_snizeni_wx = metrics['wx']['snizeni'] if 'wx' in metrics else 0.0
        brier_sknt = metrics['sknt']['brier_reg'] if 'sknt' in metrics else 0.0
        brier_vsby = metrics['vsby']['brier_reg'] if 'vsby' in metrics else 0.0
        brier_wx = metrics['wx']['brier_reg'] if 'wx' in metrics else 0.0
        accuracy_sknt = metrics['sknt']['accuracy_reg'] * 100 if 'sknt' in metrics else 0.0
        accuracy_vsby = metrics['vsby']['accuracy_reg'] * 100 if 'vsby' in metrics else 0.0
        accuracy_wx = metrics['wx']['accuracy_reg'] * 100 if 'wx' in metrics else 0.0

        summary.append({
            'Station': station,
            'Snížení SKNT (%)': round(avg_snizeni_sknt, 1),
            'Snížení VSBY (%)': round(avg_snizeni_vsby, 1),
            'Snížení WX (%)': round(avg_snizeni_wx, 1),
            'Brier SKNT': round(brier_sknt, 3),
            'Brier VSBY': round(brier_vsby, 3),
            'Brier WX': round(brier_wx, 3),
            'Accuracy SKNT (%)': round(accuracy_sknt, 1),
            'Accuracy VSBY (%)': round(accuracy_vsby, 1),
            'Accuracy WX (%)': round(accuracy_wx, 1)
        })

    summary_df = pd.DataFrame(summary)
    print('\n=== SOUHRNNÁ TABULKA PRO VŠECHNY STANICE ===')
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        output_file = 'taf_precision_summary.csv'
        summary_df.to_csv(output_file, index=False)
        print(f'\nSouhrn uložen do {output_file}')

        print('\nPrůměrné snížení přesnosti přes všechny stanice:')
        print(f"SKNT: {summary_df['Snížení SKNT (%)'].mean():.1f}%")
        print(f"VSBY: {summary_df['Snížení VSBY (%)'].mean():.1f}%")
        print(f"WX: {summary_df['Snížení WX (%)'].mean():.1f}%")

        print('\nPrůměrný Brier score (předpisový, ideální 0):')
        print(f"SKNT: {summary_df['Brier SKNT'].mean():.3f}")
        print(f"VSBY: {summary_df['Brier VSBY'].mean():.3f}")
        print(f"WX: {summary_df['Brier WX'].mean():.3f}")

        print('\nPrůměrná přesnost (Accuracy) předpisová přes všechny stanice (pro všechny TAFy):')
        print(f"SKNT: {summary_df['Accuracy SKNT (%)'].mean():.1f}%")
        print(f"VSBY: {summary_df['Accuracy VSBY (%)'].mean():.1f}%")
        print(f"WX: {summary_df['Accuracy WX (%)'].mean():.1f}%")
        print(f"Celková průměrná přesnost (všechny parametry): {summary_df[['Accuracy SKNT (%)', 'Accuracy VSBY (%)', 'Accuracy WX (%)']].mean().mean():.1f}%")
    else:
        print('Žádné stanice nebyly zpracovány (možná filtr nebo malý počet záznamů).')