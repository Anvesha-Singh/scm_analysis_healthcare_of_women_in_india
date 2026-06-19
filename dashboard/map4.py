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
OUTPUT_FILE = os.path.join("maps", "Map4_Insurance_Paradox_Dashboard.html")

# ==========================================
# 1. LOAD & PREP DATA 
# ==========================================
map_keys, india_geo, states_geo, states_gdf = load_spatial_data(TOPOJSON_FILE)
df = load_and_prep_csv(CSV_FILE, STATES_JSON, DISTRICTS_JSON, map_keys)

# ==========================================
# 2. DISTRICT-LEVEL CAUSAL OLS ENGINE 
# ==========================================
print("Calculating Hyper-Local Insurance Shield Effects...")

# Encode Outcomes & Treatments
df['barrier_money_bin'] = np.where(df['barrier_money'] == 1, 1, 0)
df['caste_reserved'] = np.where(df['caste'].isin([1, 2, 3]), 1, 0)
df['internet_binary'] = np.where(df['internet_use'].isin([1, 2, 3]), 1, 0)

# We must ensure govt_insurance is numeric 0/1 and drop missing values
df = df.dropna(subset=['govt_insurance', 'barrier_money_bin'])

categorical_cols = ['residence_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

dist_shield_coef = {}

for dist_key in df['MatchKey'].dropna().unique():
    dist_df = df[df['MatchKey'] == dist_key].copy()
    
    # Need enough respondents, AND variance in insurance (can't run if 0 people have it)
    if len(dist_df) > 150 and dist_df['govt_insurance'].nunique() > 1: 
        try:
            # Leaner formula to prevent singular matrix errors at the local level
            formula = (
                "barrier_money_bin ~ govt_insurance + caste_reserved + "
                "wealth_index + education_level + age + C(residence_type)"
            )
            model = smf.ols(formula, data=dist_df).fit()
            
            # Extract the effect of Govt Insurance on the Financial Barrier
            dist_shield_coef[dist_key] = model.params['govt_insurance']
            
        except Exception as e:
            dist_shield_coef[dist_key] = np.nan

df['insurance_shield_score'] = df['MatchKey'].map(dist_shield_coef)

# Cap the extreme outliers to keep the map readable
df['insurance_shield_score'] = df['insurance_shield_score'].clip(lower=-0.25, upper=0.25)

def categorize_shield(score):
    if pd.isna(score): return "Insufficient Data"
    elif score <= -0.05: return "Insurance Helps"
    elif -0.05 < score <= -0.01: return "Weak Insurance Help"
    elif -0.01 < score <= 0.05: return "Insurance Ineffective"
    else: return "Barrier Worsens"

df['shield_category'] = df['insurance_shield_score'].apply(categorize_shield)

# =====================================================================
# 3. AGGREGATE MAP DATA
# =====================================================================
print("Generating Map Data...")
map4_data = df.groupby('MatchKey').agg({
    'insurance_shield_score': 'first',     # Drives the map color
    'shield_category': 'first',            
    'barrier_money_bin': 'mean',           # Average barrier for context
    'govt_insurance': 'mean',              # Coverage % for context
    'state_code': 'first',                
    'district_code': 'first'                  
}).reset_index()

# Convert to percentages for the tooltip
map4_data['barrier_pct'] = map4_data['barrier_money_bin'] * 100
map4_data['coverage_pct'] = map4_data['govt_insurance'] * 100

map4_data.columns = [
    'MatchKey', 'insurance_shield_score', 'shield_category', 
    'raw_barrier', 'raw_coverage', 'state_code', 'district_code',
    'barrier_pct', 'coverage_pct'
]

# =====================================================================
# 4. GET PLOTLY DIV & RENDER CUSTOM DASHBOARD
# =====================================================================
print("Rendering Plotly Dashboard...")

hover_data = {
    "MatchKey": False, 
    "state_code": True,
    "district_code": True,
    "insurance_shield_score": ":.3f", 
    "coverage_pct": ":.1f",
    "barrier_pct": ":.1f",
    "shield_category": True
}

labels = {
    "state_code": "State Code",
    "district_code": "District Code",
    "insurance_shield_score": "Barrier Change (Coef)",
    "coverage_pct": "Local Insured Pop (%)",
    "barrier_pct": "Face Money Barrier (%)",
    "shield_category": "Efficacy Status"
}

# Apply your specific layout overrides for the diverging legend
legend_layout = {
    "coloraxis_colorbar": dict(
        title="<b>Govt Insurance<br>Efficacy in Reducing<br>Financial Barrier",
        thicknessmode="pixels", thickness=20,
        lenmode="pixels", len=400,
        yanchor="middle", y=0.5,
        xanchor="left", x=1.02,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray", borderwidth=1,
        tickvals=[-0.15, 0, 0.15],
        ticktext=["Effective (Green)", "Zero Effect", "Ineffective (Red)"]
    ),
    "margin": {"r":120,"t":40,"l":0,"b":0} 
}

# Note the RdYlGn_r scale and the color_midpoint anchoring exactly at 0
plotly_html_string = generate_plotly_map_html(
    map_data=map4_data,
    geojson=india_geo,
    states_geojson=states_geo,
    states_gdf=states_gdf,
    color_col='insurance_shield_score',
    hover_data_dict=hover_data,
    labels_dict=labels,
    map_title="Government Insurance Shield Effect on Out-of-Pocket Barriers",
    color_scale="RdYlGn_r",
    color_midpoint=0,
    custom_layout=legend_layout
)

# =====================================================================
# 5. BUILD HTML DASHBOARD
# =====================================================================
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Insurance Benefit Evaluation</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        h1 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; color: #2c3e50; }}
        .explanation-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #27ae60; }}
        .card h3 {{ margin-top: 0; color: #27ae60; }}
        .layout {{ display: flex; gap: 20px; align-items: stretch; }}
        .map-section {{ flex: 3; background: #fff; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
        .sidebar {{ flex: 1; display: flex; flex-direction: column; gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.95em; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-fatal {{ background: #c82333; color: #ffffff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Does Government Insurance Actually Help the People who Need it?</h1>
        <div class="layout">
            <div class="map-section">
                {plotly_html_string}
            </div>
            
            <div class="sidebar">
                <div style="background: white; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                    <h3 style="margin-top:0; color:#2c3e50;">Tooltip Logic: The Shield Score</h3>
                    <p style="font-size: 0.9em; color: #666;">This score measures the <i>reduction in probability</i> of facing a financial barrier due to having government insurance. A <strong>negative coefficient is good</strong> (insurance subtracts the barrier).</p>
                    <table>
                        <tr>
                            <th>Category</th>
                            <th>Meaning</th>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Insurance Helps</span></td>
                            <td>Negative Score. Having insurance meaningfully drops out-of-pocket costs and financial barriers in this district.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-warning">Weak Insurance Help</span></td>
                            <td>Near-zero Score. Insurance helps slightly, but structural costs remain high.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-danger">Insurance Ineffective</span></td>
                            <td>Zero or positive. Having insurance provides virtually no financial relief; patients still report critical money barriers.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-fatal">Barrier Worsens</span></td>
                            <td>Highly Positive. Insured individuals are actually <i>more</i> likely to face high costs, suggesting systemic exploitation or out-of-pocket upcharges.</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        <div class="explanation-grid">
            <div class="card">
                <h3>1. What is this map?</h3>
                <p>This map identifies exactly where government health insurance schemes (like Ayushman Bharat) are functioning as intended, and where they exist purely "on paper" while failing to protect citizens financially.</p>
            </div>
            <div class="card">
                <h3>2. How is the causal math used?</h3>
                <p>We ran an OLS regression controlling for Wealth, Caste, Education, and Age. We specifically isolated the effect of <i>possessing government insurance</i> on a person reporting <i>"Money is a big problem"</i> when seeking care. A successful scheme yields a negative causal coefficient.</p>
            </div>
            <div class="card">
                <h3>3. How to interpret the results?</h3>
                <p><strong>Green districts (Negative Coef)</strong> are policy success stories. The infrastructure accepts the insurance and waives fees. <strong>Red districts (Positive Coef)</strong> are systemic failures. In these red zones, giving a poor family an insurance card does not stop them from paying out-of-pocket.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Dashboard successfully generated! Open {OUTPUT_FILE} to view.")