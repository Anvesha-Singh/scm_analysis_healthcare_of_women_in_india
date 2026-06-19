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
OUTPUT_FILE = os.path.join("maps", "Map1_Systemic_Infrastructure_Dashboard.html")

# ==========================================
# 1. LOAD & PREP DATA 
# ==========================================
map_keys, india_geo, states_geo, states_gdf = load_spatial_data(TOPOJSON_FILE)
df = load_and_prep_csv(CSV_FILE, STATES_JSON, DISTRICTS_JSON, map_keys)

# ==========================================
# 2. CAUSAL FEATURE ENGINEERING
# ==========================================
print("Calculating Causal Effects...")
df['distance_problem_binary'] = np.where(df['barrier_distance'] == 1, 1, 0)

categorical_cols = ['StateName', 'residence_type', 'barrier_transport', 'barrier_money']
for col in categorical_cols:
    df[col] = df[col].astype('category')

state_wealth_effect = {}

for state_name in df['StateName'].dropna().unique():
    state_df = df[df['StateName'] == state_name].copy()
    
    if len(state_df) > 50:
        try:
            formula_direct = (
                "distance_problem_binary ~ wealth_index + barrier_transport + "
                "residence_type + education_level + barrier_money"
            )
            model = smf.ols(formula_direct, data=state_df).fit()
            state_wealth_effect[state_name] = model.params['wealth_index']
        except Exception as e:
            state_wealth_effect[state_name] = np.nan

df['wealth_mitigation_effect'] = df['StateName'].map(state_wealth_effect)

def categorize_effect(effect):
    if pd.isna(effect):
        return "Insufficient Data"
    elif effect <= -0.05:
        return "Wealth Solves Issue"
    elif -0.05 < effect <= -0.01:
        return "Wealth Might Solve Issue"
    else:
        return "Wealth Doesn't Solve Issue"

df['wealth_effect_category'] = df['wealth_mitigation_effect'].apply(categorize_effect)

# =====================================================================
# 3. AGGREGATE MAP DATA
# =====================================================================
print("Generating Map Data...")
map1_data = df.groupby('MatchKey').agg({
    'distance_problem_binary': 'mean',          
    'wealth_effect_category': 'first',
    'wealth_mitigation_effect': 'first',  
    'state_code': 'first',                
    'district_code': 'first'                  
}).reset_index()

map1_data.columns = [
    'MatchKey', 
    'overall_infra_deficit', 
    'wealth_effect_category',
    'raw_wealth_effect',
    'state_code',
    'district_code'
]

# =====================================================================
# 4. GET PLOTLY DIV & RENDER CUSTOM DASHBOARD
# =====================================================================
hover_data = {
    "MatchKey": False, 
    "state_code": True,
    "district_code": True,
    "overall_infra_deficit": ":.2f",
    "wealth_effect_category": True,
    "raw_wealth_effect": ":.5f" 
}

labels = { 
    "state_code": "State Code",
    "district_code": "District Code",
    "overall_infra_deficit": "Infra Deficit (%)",
    "wealth_effect_category": "Wealth Impact",
    "raw_wealth_effect": "Causal Coef"
}

plotly_html_string = generate_plotly_map_html(
    map_data=map1_data,
    geojson=india_geo,
    states_geojson=states_geo,
    states_gdf=states_gdf,
    color_col='overall_infra_deficit',
    hover_data_dict=hover_data,
    labels_dict=labels,
    map_title="Systemic Infrastructure Deficit (Distance Barrier Across All Wealth Brackets)"
)

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Healthcare Infrastructure Deficit Map</title>
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
        .badge-fail {{ background: #ffcccc; color: #cc0000; }}
        .badge-mid {{ background: #fff3cd; color: #856404; }}
        .badge-pass {{ background: #d4edda; color: #155724; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Does Wealth Solve the Distance Barrier to Healthcare?</h1>
        <div class="layout">
            <div class="map-section">
                {plotly_html_string}
            </div>
            
            <div class="sidebar">
                <div style="background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3 style="margin-top:0; color:#2c3e50;">Reference: Wealth Effect Categories</h3>
                    <p style="font-size: 0.9em; color: #666;">Derived from the causal coefficient of the wealth variable in our regression model.</p>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Coefficient Limit</th>
                            <th>Meaning</th>
                        </tr>
                        <tr>
                            <td><span class="badge badge-pass">Wealth Solves Issue</span></td>
                            <td>Less than -0.05</td>
                            <td>Moving up a wealth bracket noticeably reduces the distance barrier (>5% reduction).</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-mid">Wealth Might Solve Issue</span></td>
                            <td>-0.05 to -0.01</td>
                            <td>Wealth provides a marginal advantage (0.5-5% reduction in barrier probability).</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-fail">Wealth Doesn't Solve Issue</span></td>
                            <td>Greater than -0.01</td>
                            <td>Wealth provides virtually zero advantage (<1% reduction). The problem is entirely structural and geographical.</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <div class="explanation-grid">
            <div class="card">
                <h3>1. What is this map?</h3>
                <p>This map visualizes the <strong>pure physical infrastructure deficit</strong> across Indian districts. The colors represent the percentage of the overall population who report that physical distance is a major barrier to accessing healthcare. <strong>Darker red indicates a severe lack of nearby facilities.</strong></p>
            </div>
            <div class="card">
                <h3>2. How was this calculated?</h3>
                <p>We used a <strong>Causal Inference OLS Model</strong> to isolate the effect of personal Wealth on the Distance Barrier. By strictly controlling for Transport availability, Rural/Urban residence, and Money barriers, we mathematically isolate "pure physical geography" from a person's "ability to travel."</p>
            </div>
            <div class="card">
                <h3>3. How to interpret the results?</h3>
                <p>Hover over any district. You will see the severity of the deficit alongside the <strong>Wealth Effect</strong> category. If a region is Dark Red AND says "Wealth Doesn't Solve Issue", it signifies a systemic infrastructure failure where personal finances cannot overcome the lack of physical clinics.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Dashboard successfully generated! Open {OUTPUT_FILE} to view.")