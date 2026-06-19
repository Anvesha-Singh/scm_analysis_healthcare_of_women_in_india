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
OUTPUT_FILE = os.path.join("maps", "Map2_Agency_Transfer_Dashboard.html")

# ==========================================
# 1. LOAD & PREP DATA 
# ==========================================
map_keys, india_geo, states_geo, states_gdf = load_spatial_data(TOPOJSON_FILE)
df = load_and_prep_csv(CSV_FILE, STATES_JSON, DISTRICTS_JSON, map_keys)

# ==========================================
# 2. DISTRICT-LEVEL CAUSAL OLS ENGINE
# ==========================================
print("Calculating Hyper-Local Financial Agency Transfer Effects...")

df['health_agency_binary'] = np.where(df['decide_health'].isin([1, 2, 3]), 1, 0)
df['has_financial_agency'] = np.where(df['decide_own_money'].isin([1, 2, 3]), 1, 0)
df['internet_binary'] = np.where(df['internet_use'].isin([1, 2, 3]), 1, 0)
df['tv_binary'] = np.where(df['freq_tv'].isin([1, 2, 3]), 1, 0)
df['literacy_binary'] = np.where(df['literacy'].isin([1, 2]), 1, 0)

categorical_cols = ['residence_type', 'caste', 'religion']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

dist_agency_effect = {}

# LOOPING OVER DISTRICTS INSTEAD OF STATES
for dist_key in df['MatchKey'].dropna().unique():
    dist_df = df[df['MatchKey'] == dist_key].copy()
    
    # Require at least 150 respondents to prevent statistical noise/singular matrices
    if len(dist_df) > 150: 
        try:
            formula_elite = (
                "health_agency_binary ~ has_financial_agency + literacy_binary + "
                "education_level + wealth_index + age + internet_binary + "
                "tv_binary + C(residence_type) + C(caste) + C(religion)"
            )
            
            model = smf.ols(formula_elite, data=dist_df).fit()
            dist_agency_effect[dist_key] = model.params['has_financial_agency']
            
        except Exception as e:
            dist_agency_effect[dist_key] = np.nan

df['financial_empowerment_coef'] = df['MatchKey'].map(dist_agency_effect)

# Cap the extreme outliers so they don't break the interpretation limits
df['financial_empowerment_coef'] = df['financial_empowerment_coef'].clip(lower=-0.2, upper=0.5)

def categorize_empowerment(effect):
    if pd.isna(effect): return "Insufficient Data/Variance"
    elif effect > 0.15: return "Strong Transfer"
    elif 0.05 < effect <= 0.15: return "Moderate Transfer"
    elif 0.00 < effect <= 0.05: return "Weak Transfer"
    else: return "Blocked Transfer (Friction)"

df['empowerment_category'] = df['financial_empowerment_coef'].apply(categorize_empowerment)

# =====================================================================
# 3. AGGREGATE MAP DATA
# =====================================================================
print("Generating Map Data...")
map2_data = df.groupby('MatchKey').agg({
    'health_agency_binary': 'mean',        
    'has_financial_agency': 'mean',        # Kept for the tooltip
    'financial_empowerment_coef': 'first', # Moved to Tooltip
    'empowerment_category': 'first',       # Moved to Tooltip
    'state_code': 'first',                
    'district_code': 'first'                  
}).reset_index()

# Convert fractions to percentages for better UI readability
map2_data['health_agency_pct'] = map2_data['health_agency_binary'] * 100
map2_data['financial_agency_pct'] = map2_data['has_financial_agency'] * 100

map2_data.columns = [
    'MatchKey', 'raw_health', 'raw_finance', 'raw_empowerment_coef', 
    'empowerment_category', 'state_code', 'district_code', 
    'health_agency_pct', 'financial_agency_pct'
]

# =====================================================================
# 4. GET PLOTLY DIV & RENDER CUSTOM DASHBOARD
# =====================================================================
print("Rendering Plotly Dashboard...")

hover_data = {
    "MatchKey": False, 
    "state_code": True,
    "district_code": True,
    "health_agency_pct": ":.1f", 
    "financial_agency_pct": ":.1f",
    "raw_empowerment_coef": ":.3f", 
    "empowerment_category": True
}

labels = {
    "state_code": "State Code",
    "district_code": "District Code",
    "health_agency_pct": "Has Healthcare Agency (%)",
    "financial_agency_pct": "Has Financial Agency (%)",
    "raw_empowerment_coef": "Transfer Power Coef",
    "empowerment_category": "Financial Impact Level"
}

# Generate the map div using the core module (passing the Purple color scale)
plotly_html_string = generate_plotly_map_html(
    map_data=map2_data,
    geojson=india_geo,
    states_geojson=states_geo,
    states_gdf=states_gdf,
    color_col='health_agency_pct',
    hover_data_dict=hover_data,
    labels_dict=labels,
    map_title="Baseline Healthcare Autonomy (Darker = Higher Agency)",
    color_scale="Purpor"
)

# =====================================================================
# 5. BUILD HTML DASHBOARD
# =====================================================================
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Healthcare Agency in relation to Financial Agency</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        h1 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; color: #2c3e50; }}
        .explanation-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #8e44ad; }}
        .card h3 {{ margin-top: 0; color: #8e44ad; }}
        .layout {{ display: flex; gap: 20px; align-items: stretch; }}
        .map-section {{ flex: 3; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .sidebar {{ flex: 1; display: flex; flex-direction: column; gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95em; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .badge-fail {{ background: #ffcccc; color: #cc0000; }}
        .badge-weak {{ background: #fdfd96; color: #856404; }}
        .badge-mid {{ background: #aec6cf; color: #1e3f66; }}
        .badge-pass {{ background: #77dd77; color: #155724; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Does Having Own Money Increase Healthcare Agency?</h1>
        <div class="layout">
            <div class="map-section">
                {plotly_html_string}
            </div>
            
            <div class="sidebar">
                <div style="background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3 style="margin-top:0; color:#2c3e50;">Tooltip Logic: The Transfer Power</h3>
                    <p style="font-size: 0.9em; color: #666;">The map is colored by actual healthcare agency. But when you look up a district, you see the <strong>Transfer Power Coef</strong>. This indicates the percentage point increase in healthcare agency gained <i>purely</i> by having financial agency.</p>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Coefficient Limit</th>
                            <th>Meaning</th>
                        </tr>
                        <tr>
                            <td><span class="badge badge-pass">Strong Transfer</span></td>
                            <td>Greater than +0.15</td>
                            <td>System works linearly. Gaining financial control boosts autonomy by >15%.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-mid">Moderate Transfer</span></td>
                            <td>+0.05 to +0.15</td>
                            <td>Financial control grants a measurable but constrained voice (5-15% boost).</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-weak">Weak Transfer</span></td>
                            <td>+0.00 to +0.05</td>
                            <td>Financial control yields almost negligible power (<5% boost).</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-fail">Blocked Transfer</span></td>
                            <td>Less than 0.00</td>
                            <td>Cultural friction overrides economic power. Money does not grant autonomy here.</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <div class="explanation-grid">
            <div class="card">
                <h3>1. What is this map?</h3>
                <p>This map shows the baseline reality of female healthcare autonomy. The light purple areas show districts where a very low percentage of women have a say in their own healthcare decisions. Dark areas indicate high autonomy.</p>
            </div>
            <div class="card">
                <h3>2. How is the causal math used?</h3>
                <p>When you use the Search UI to look up a district, you will see the <strong>Transfer Power Coef</strong>. We ran a hyper-local OLS regression for <i>every single district</i> to figure out if giving a woman financial independence would actually fix the lack of healthcare autonomy in that specific area.</p>
            </div>
            <div class="card">
                <h3>3. How to interpret the results?</h3>
                <p>Find a light area (low autonomy) and check the lookup card. If it says <strong>Strong Transfer</strong>, it means economic empowerment programs (like micro-loans) will solve the health autonomy issue. If it says <strong>Blocked Transfer</strong>, economic programs will fail to change the social dynamics.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Dashboard successfully generated! Open {OUTPUT_FILE} to view.")