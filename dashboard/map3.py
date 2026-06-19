import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

from map_core import load_spatial_data, load_and_prep_csv, generate_plotly_map_html

# ==========================================
# 0. CONFIGURATION & FILE PATHS
# ==========================================
TOPOJSON_FILE = 'india-districts-2019-734.json'
CSV_FILE = 'india_nfhs5_cleaned.csv'
STATES_JSON = 'states.json'
DISTRICTS_JSON = 'districts.json'

os.makedirs('maps', exist_ok=True)
OUTPUT_FILE = os.path.join("maps", "Map3_TV_vs_Education_Dashboard.html")

# ==========================================
# 1. LOAD & PREP DATA 
# ==========================================
map_keys, india_geo, states_geo, states_gdf = load_spatial_data(TOPOJSON_FILE)
df = load_and_prep_csv(CSV_FILE, STATES_JSON, DISTRICTS_JSON, map_keys)

# ==========================================
# 2. DISTRICT-LEVEL CAUSAL OLS ENGINE (TV VS EDU)
# ==========================================
print("Calculating Hyper-Local TV vs Education Intensity Effects...")

df['health_agency_binary'] = np.where(df['decide_health'].isin([1, 2, 3]), 1, 0)

categorical_cols = ['residence_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

dist_tv_advantage = {}
dist_tv_coef = {}
dist_edu_coef = {}

for dist_key in df['MatchKey'].dropna().unique():
    dist_df = df[df['MatchKey'] == dist_key].copy()
    
    # Drop rows missing specific variables to ensure clean Z-scores
    dist_df = dist_df.dropna(subset=['freq_tv', 'education_level'])
    
    if len(dist_df) > 150: 
        try:
            # Standardize (Z-Score) the raw ordinal variables directly
            dist_df['tv_z'] = (dist_df['freq_tv'] - dist_df['freq_tv'].mean()) / (dist_df['freq_tv'].std() + 1e-9)
            dist_df['edu_z'] = (dist_df['education_level'] - dist_df['education_level'].mean()) / (dist_df['education_level'].std() + 1e-9)
            
            formula = "health_agency_binary ~ tv_z + edu_z + wealth_index + age + C(residence_type)"
            model = smf.ols(formula, data=dist_df).fit()
            
            t_coef = model.params['tv_z']
            e_coef = model.params['edu_z']
            
            dist_tv_coef[dist_key] = t_coef
            dist_edu_coef[dist_key] = e_coef
            
            # Calculate Dominance (Positive = TV Wins, Negative = Edu Wins)
            dist_tv_advantage[dist_key] = t_coef - e_coef
            
        except Exception as e:
            dist_tv_advantage[dist_key] = np.nan

df['tv_advantage_score'] = df['MatchKey'].map(dist_tv_advantage)
df['raw_tv_coef'] = df['MatchKey'].map(dist_tv_coef)
df['raw_edu_coef'] = df['MatchKey'].map(dist_edu_coef)

# Cap the extreme outliers to keep the gradient map colors balanced
df['tv_advantage_score'] = df['tv_advantage_score'].clip(lower=-0.15, upper=0.15)

def categorize_driver(score):
    if pd.isna(score): return "Insufficient Data"
    elif score > 0.04: return "Strong TV Association"
    elif 0.01 < score <= 0.04: return "Slight TV Edge"
    elif -0.01 <= score <= 0.01: return "Equal Impact"
    elif -0.04 <= score < -0.01: return "Slight Education Edge"
    else: return "Strong Education Association"

df['driver_category'] = df['tv_advantage_score'].apply(categorize_driver)

# =====================================================================
# 3. AGGREGATE MAP DATA
# =====================================================================
print("Generating Map Data...")
map3_data = df.groupby('MatchKey').agg({
    'tv_advantage_score': 'first', 
    'driver_category': 'first',
    'raw_tv_coef': 'first',
    'raw_edu_coef': 'first',
    'state_code': 'first',                
    'district_code': 'first'                  
}).reset_index()

# =====================================================================
# 4. GET PLOTLY DIV & RENDER CUSTOM DASHBOARD
# =====================================================================
print("Rendering Plotly Dashboard...")

hover_data = {
    "MatchKey": False, 
    "state_code": True,
    "district_code": True,
    "tv_advantage_score": ":.3f", 
    "raw_tv_coef": ":.3f",
    "raw_edu_coef": ":.3f",
    "driver_category": True
}

labels = {
    "state_code": "State Code",
    "district_code": "District Code",
    "tv_advantage_score": "TV Advantage Margin",
    "raw_tv_coef": "TV Strength (Z)",
    "raw_edu_coef": "Education Strength (Z)",
    "driver_category": "Dominant Driver"
}

# The specific colorbar overrides for the custom legend
legend_layout = {
    "coloraxis_colorbar": dict(
        title="<b>Dominant Driver</b><br>TV (red) vs Education (blue)<br>",
        thicknessmode="pixels", thickness=20,
        lenmode="pixels", len=400,
        yanchor="middle", y=0.5,
        xanchor="left", x=1.02,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray", borderwidth=1,
        tickvals=[-0.1, 0, 0.1],
        ticktext=["Edu Wins", "Tie", "TV Wins"]
    ),
    "margin": {"r":120,"t":40,"l":0,"b":0} 
}

plotly_html_string = generate_plotly_map_html(
    map_data=map3_data,
    geojson=india_geo,
    states_geojson=states_geo,
    states_gdf=states_gdf,
    color_col='tv_advantage_score',
    hover_data_dict=hover_data,
    labels_dict=labels,
    map_title="Impact of Television vs. Formal Education on Autonomy",
    color_scale="RdBu_r",
    color_midpoint=0,            # Locks white perfectly to zero
    custom_layout=legend_layout  # Injects the custom side-bar legend
)

# =====================================================================
# 5. BUILD HTML DASHBOARD
# =====================================================================
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Media Influence on Healthcare Autonomy</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        h1 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; color: #2c3e50; }}
        .explanation-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #e74c3c; }}
        .card h3 {{ margin-top: 0; color: #e74c3c; }}
        .layout {{ display: flex; gap: 20px; align-items: stretch; }}
        .map-section {{ flex: 3; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .sidebar {{ flex: 1; display: flex; flex-direction: column; gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95em; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .badge-tv-strong {{ background: #ffcccc; color: #cc0000; }}
        .badge-tv-slight {{ background: #ffe5e5; color: #a30000; }}
        .badge-tie {{ background: #eeeeee; color: #555555; }}
        .badge-edu-slight {{ background: #cce5ff; color: #004085; }}
        .badge-edu-strong {{ background: #99ccff; color: #002752; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Impact of Television vs. Formal Education on Healthcare Autonomy</h1>
        <div class="layout">
            <div class="map-section">
                {plotly_html_string}
            </div>
            
            <div class="sidebar">
                <div style="background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3 style="margin-top:0; color:#2c3e50;">Tooltip Logic: The Association Score</h3>
                    <p style="font-size: 0.9em; color: #666;">This map compares two "dosages": frequency of TV consumption vs. years of formal schooling. Red areas mean the habit of watching TV grants a stronger boost to autonomy than formal education. Blue means Education is the superior driver.</p>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Meaning</th>
                        </tr>
                        <tr>
                            <td><span class="badge badge-tv-strong">Strong TV Dominance</span></td>
                            <td>Increasing daily TV viewership heavily outweighs advancing formal schooling for gaining a voice.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-tv-slight">Slight TV Edge</span></td>
                            <td>TV consumption provides a moderately stronger statistical boost to autonomy than additional formal schooling.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-tie">Equal Impact</span></td>
                            <td>Both institutions (broadcast media and schooling) provide roughly the same boost to autonomy per "unit" of intensity.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-edu-slight">Slight Education Edge</span></td>
                            <td>Advancing formal education provides a moderately stronger boost to autonomy than daily TV viewership.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-edu-strong">Strong Education Dominance</span></td>
                            <td>TV has little effect; advancing through formal education is the strict gatekeeper to personal empowerment here.</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <div class="explanation-grid">
            <div class="card">
                <h3>1. What is this map?</h3>
                <p>This is a <strong>Causal Battleground Map</strong>. Instead of just looking at "Yes/No" access, it compares the intensity of consumption. It answers: <i>Does the daily habit of consuming broadcast media reshape cultural norms faster than years spent in a classroom?</i></p>
            </div>
            <div class="card">
                <h3>2. How is the causal math used?</h3>
                <p>Inside every district, we standard-scaled (Z-scored) the TV frequency scale (0-3) and the Education level scale (0-5). This created perfectly fair "Beta Weights". We then subtracted the Education coefficient from the TV coefficient. A positive result (Red) means TV won. A negative result (Blue) means Education won.</p>
            </div>
            <div class="card">
                <h3>3. How to interpret the results?</h3>
                <p>If you see a cluster of Red districts, broadcast media is a highly effective vehicle for social change and shifting gender norms. If you see Blue, the local culture likely demands formal, traditional degrees to grant women a voice, rendering passive media consumption ineffective.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Dashboard successfully generated! Open {OUTPUT_FILE} to view.")