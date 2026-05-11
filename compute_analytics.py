import pandas as pd
import numpy as np
import json
import math
import sqlite3

FILE_PATHS = ["elastic1.csv", "elastic2.csv"]

print("Loading CSVs...")
dfs = [pd.read_csv(p, low_memory=False) for p in FILE_PATHS]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows")

# ── Timestamp parsing ──────────────────────────────────────────────
clean_ts = df['@timestamp'].str.replace(' @ ', ' ')
clean_lt = df['original.time'].str.replace(' @ ', ' ')
df['@timestamp'] = pd.to_datetime(clean_ts, errors='coerce')
df['original.time'] = pd.to_datetime(clean_lt, errors='coerce')
df = df.dropna(subset=['@timestamp'])
print(f"After timestamp parse: {len(df):,} rows")

df['date'] = df['@timestamp'].dt.date
df['device_type'] = pd.to_numeric(df['device_type'], errors='coerce')

# ── Activation dates ───────────────────────────────────────────────
activation_dates = df.groupby('hostname')['date'].min().reset_index()
activation_dates.columns = ['hostname', 'activation_date']
df = df.merge(activation_dates, on='hostname', how='left')
df['date'] = pd.to_datetime(df['date'])
df['activation_date'] = pd.to_datetime(df['activation_date'])
df['days_since_activation'] = (df['date'] - df['activation_date']).dt.days

# ── GAME_START subset ─────────────────────────────────────────────
df_games = df[df['event'] == 'GAME_START'].copy()
df_games['local_hour'] = df_games['original.time'].dt.hour
df_games = df_games[df_games['geoip.country_code'].notna()]
df_games_clean = df_games[df_games['geoip.country_code'] != '-'].copy()

# ─────────────────────────────────────────────────────────────────
# 1. Hourly usage (global)
# ─────────────────────────────────────────────────────────────────
hourly = df_games['local_hour'].value_counts().sort_index()
hourly_data = [{"hour": int(h), "plays": int(c)} for h, c in hourly.items()]

# ─────────────────────────────────────────────────────────────────
# 2. Country hourly usage
# ─────────────────────────────────────────────────────────────────
country_hourly_raw = (
    df_games_clean
    .groupby(['geoip.country_code', 'local_hour'])
    .size()
    .reset_index(name='plays')
)
country_hourly_raw['local_hour'] = country_hourly_raw['local_hour'].astype(int)
country_hourly_raw = country_hourly_raw[
    (country_hourly_raw['local_hour'] >= 0) & (country_hourly_raw['local_hour'] <= 23)
]

country_hourly_data = {}
for country, grp in country_hourly_raw.groupby('geoip.country_code'):
    row = [0] * 24
    for _, r in grp.iterrows():
        row[int(r['local_hour'])] = int(r['plays'])
    country_hourly_data[country] = row

# ─────────────────────────────────────────────────────────────────
# 3. Seasonal trends
# ─────────────────────────────────────────────────────────────────
month_to_season = {
    1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring', 6:'Summer',
    7:'Summer', 8:'Summer', 9:'Autumn', 10:'Autumn', 11:'Autumn', 12:'Winter'
}
df_games['season'] = df_games['@timestamp'].dt.month.map(month_to_season)
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
seasonal_counts = df_games['season'].value_counts()
seasonal_data = [{"season": s, "plays": int(seasonal_counts.get(s, 0))} for s in season_order]

# ─────────────────────────────────────────────────────────────────
# 4. Session durations
# ─────────────────────────────────────────────────────────────────
df_sorted = df.sort_values(['game_uuid', '@timestamp'])
starts_ends = df_sorted[df_sorted['event'].isin(['GAME_START', 'GAME_END'])].copy()
starts_ends['next_event'] = starts_ends.groupby('game_uuid')['event'].shift(-1)
starts_ends['next_timestamp'] = starts_ends.groupby('game_uuid')['@timestamp'].shift(-1)
valid_sessions = starts_ends[
    (starts_ends['event'] == 'GAME_START') & (starts_ends['next_event'] == 'GAME_END')
].copy()
valid_sessions['duration_minutes'] = (
    valid_sessions['next_timestamp'] - valid_sessions['@timestamp']
).dt.total_seconds() / 60

df_sessions = valid_sessions[['game_uuid', 'generic_string', '@timestamp', 'next_timestamp', 'duration_minutes']].copy()

global_stats = df_sessions.groupby('generic_string').agg(
    total_launches=('game_uuid', 'count'),
    avg_duration=('duration_minutes', 'mean'),
    total_duration=('duration_minutes', 'sum')
).reset_index()

top_launches = global_stats.nlargest(10, 'total_launches')
top_avg = global_stats[global_stats['total_launches'] >= 5].nlargest(10, 'avg_duration')
top_cumulative = global_stats.nlargest(10, 'total_duration')

def to_records(df_in, name_col, val_cols):
    return [
        {**{"name": str(r[name_col])}, **{c: round(float(r[c]), 2) for c in val_cols}}
        for _, r in df_in.iterrows()
    ]

sessions_data = {
    "top_launches": to_records(top_launches, 'generic_string', ['total_launches']),
    "top_avg_duration": to_records(top_avg, 'generic_string', ['avg_duration']),
    "top_cumulative": to_records(top_cumulative, 'generic_string', ['total_duration'])
}

# All games (≥5 sessions) with avg_duration + launches for scatter chart
all_games = global_stats[global_stats['total_launches'] >= 5].copy()
all_games['avg_duration'] = all_games['avg_duration'].round(2)
all_game_stats = [
    {"name": str(r['generic_string']), "launches": int(r['total_launches']), "avg_duration": float(r['avg_duration'])}
    for _, r in all_games.iterrows()
]

# ─────────────────────────────────────────────────────────────────
# 5. Per-game detailed stats (country, hourly, seasonal)
# ─────────────────────────────────────────────────────────────────
game_country_data = {}
game_hourly_data = {}
game_seasonal_data = {}

for game_name, grp in df_games.groupby('generic_string'):
    key = str(game_name)
    cc = grp[grp['geoip.country_code'] != '-']['geoip.country_code'].value_counts()
    game_country_data[key] = {str(k): int(v) for k, v in cc.items()}
    row = [0] * 24
    for h, cnt in grp['local_hour'].dropna().value_counts().items():
        h = int(h)
        if 0 <= h <= 23:
            row[h] = int(cnt)
    game_hourly_data[key] = row
    game_seasonal_data[key] = {str(s): int(c) for s, c in grp['season'].value_counts().items()}

# ─────────────────────────────────────────────────────────────────
# 6. Per-app-uuid stats (top 300 by launch count)
# ─────────────────────────────────────────────────────────────────
app_uuid_counts = df_games['app_uuid'].value_counts()
print(f"  Unique app_uuids: {len(app_uuid_counts)}")
top_uuids = app_uuid_counts.head(300).index

app_stats = {}
for uuid in top_uuids:
    grp = df_games[df_games['app_uuid'] == uuid]
    game_name = str(grp['generic_string'].mode().iloc[0])
    cc = grp[grp['geoip.country_code'] != '-']['geoip.country_code'].value_counts()
    countries = {str(k): int(v) for k, v in cc.items()}
    row = [0] * 24
    for h, cnt in grp['local_hour'].dropna().value_counts().items():
        h = int(h)
        if 0 <= h <= 23:
            row[h] = int(cnt)
    app_stats[str(uuid)] = {
        'game': game_name,
        'launches': int(len(grp)),
        'countries': countries,
        'hourly': row
    }

game_detail = {
    'country': game_country_data,
    'hourly': game_hourly_data,
    'seasonal': game_seasonal_data
}
with open('game_data.json', 'w') as f:
    json.dump(game_detail, f, separators=(',', ':'))
with open('app_data.json', 'w') as f:
    json.dump(app_stats, f, separators=(',', ':'))

# ─────────────────────────────────────────────────────────────────
# 7. Retention curve
# ─────────────────────────────────────────────────────────────────
daily_retention = (
    df.groupby('days_since_activation')['hostname']
    .nunique()
    .reset_index(name='unique_devices')
    .sort_values('days_since_activation')
)
retention_plot = daily_retention[
    (daily_retention['days_since_activation'] >= 5) &
    (daily_retention['days_since_activation'] <= 500)
]
retention_data = [
    {"day": int(r['days_since_activation']), "devices": int(r['unique_devices'])}
    for _, r in retention_plot.iterrows()
]

# ─────────────────────────────────────────────────────────────────
# 8. Games by device type
# ─────────────────────────────────────────────────────────────────
games_per_device = (
    df_games.groupby(['device_type', 'generic_string'])
    .size()
    .reset_index(name='play_count')
    .sort_values(['device_type', 'play_count'], ascending=[True, False])
)
top_per_device = games_per_device.groupby('device_type').head(10)
device_data = {}
for device, grp in top_per_device.groupby('device_type'):
    device_data[str(int(device))] = [
        {"name": str(r['generic_string']), "plays": int(r['play_count'])}
        for _, r in grp.iterrows()
    ]

# ─────────────────────────────────────────────────────────────────
# 9. Top 5 games per country
# ─────────────────────────────────────────────────────────────────
country_games = (
    df_games_clean.groupby(['geoip.country_code', 'generic_string'])
    .size()
    .reset_index(name='play_count')
    .sort_values(['geoip.country_code', 'play_count'], ascending=[True, False])
)
top5_per_country = country_games.groupby('geoip.country_code').head(5)
country_data = {}
for country, grp in top5_per_country.groupby('geoip.country_code'):
    country_data[str(country)] = [
        {"name": str(r['generic_string']), "plays": int(r['play_count'])}
        for _, r in grp.iterrows()
    ]

# ─────────────────────────────────────────────────────────────────
# 10. Average session duration by country
# ─────────────────────────────────────────────────────────────────
country_dur_raw = (
    valid_sessions[
        valid_sessions['geoip.country_code'].notna() &
        (valid_sessions['geoip.country_code'] != '-')
    ]
    .groupby('geoip.country_code')['duration_minutes']
    .agg(avg_duration='mean', sessions='count')
    .reset_index()
    .sort_values('avg_duration', ascending=False)
)
avg_session_by_country = {
    str(r['geoip.country_code']): {
        "avg_duration": round(float(r['avg_duration']), 2),
        "sessions": int(r['sessions'])
    }
    for _, r in country_dur_raw.iterrows()
}

# ─────────────────────────────────────────────────────────────────
# 11. KPI summary
# ─────────────────────────────────────────────────────────────────
total_sessions = len(valid_sessions)
unique_devices = df['hostname'].nunique()
unique_countries = df_games_clean['geoip.country_code'].nunique()
total_games_played = len(df_games)
avg_session_min = round(float(valid_sessions['duration_minutes'].median()), 1)
top_game = df_games['generic_string'].value_counts().idxmax()

overall_avg_min = round(float(valid_sessions['duration_minutes'].mean()), 2)

kpi_data = {
    "total_sessions": int(total_sessions),
    "unique_devices": int(unique_devices),
    "unique_countries": int(unique_countries),
    "total_game_plays": int(total_games_played),
    "median_session_min": avg_session_min,
    "avg_session_min": overall_avg_min,
    "top_game": str(top_game)
}

# ─────────────────────────────────────────────────────────────────
# Bundle and write
# ─────────────────────────────────────────────────────────────────
analytics = {
    "kpi": kpi_data,
    "hourly_usage": hourly_data,
    "country_hourly": country_hourly_data,
    "seasonal": seasonal_data,
    "sessions": sessions_data,
    "retention": retention_data,
    "games_by_device": device_data,
    "games_by_country": country_data,
    "avg_session_by_country": avg_session_by_country,
    "all_game_stats": all_game_stats,
}

with open("analytics_data.json", "w") as f:
    json.dump(analytics, f, indent=2)

# ── Write to SQLite ────────────────────────────────────────────────────────
conn = sqlite3.connect('tover.db')
conn.execute('CREATE TABLE IF NOT EXISTS analytics (key TEXT PRIMARY KEY, value TEXT)')
for key, value in analytics.items():
    conn.execute('INSERT OR REPLACE INTO analytics (key, value) VALUES (?, ?)',
                 (key, json.dumps(value, separators=(',', ':'))))
conn.execute('INSERT OR REPLACE INTO analytics (key, value) VALUES (?, ?)',
             ('game_data', json.dumps(game_detail, separators=(',', ':'))))
conn.commit()
conn.close()

print("Done — analytics_data.json and tover.db written.")
print(f"  KPIs: {kpi_data}")
