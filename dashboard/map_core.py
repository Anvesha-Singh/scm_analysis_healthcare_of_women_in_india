import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
import re

def load_spatial_data(topojson_file):
    print("Loading TopoJSON boundary data...")
    gdf = gpd.read_file(topojson_file)
    gdf['geometry'] = gdf['geometry'].make_valid()
    gdf['geometry'] = gdf['geometry'].buffer(0)

    gdf['st_nm_clean'] = gdf['st_nm'].str.strip().str.title()
    gdf['district_clean'] = gdf['district'].str.strip().str.title()
    gdf['MatchKey'] = gdf['district_clean'] + ", " + gdf['st_nm_clean']

    india_geojson = json.loads(gdf.to_json())

    print("Generating state boundaries overlay (Dissolving)...")
    states_gdf = gdf.dissolve(by='st_nm_clean', as_index=False)
    states_geojson = json.loads(states_gdf.to_json())
    
    map_keys = set(gdf['MatchKey'].unique())

    return map_keys, india_geojson, states_geojson, states_gdf


def standardize_names(dist_name, state_name):
    # 1. Force string conversion and strip external padding
    clean_dist = str(dist_name).strip()
    clean_state = str(state_name).strip()
    
    # 2. Collapse all hidden double-spaces/tabs into a single standard space
    clean_dist = re.sub(r'\s+', ' ', clean_dist).title()
    clean_state = re.sub(r'\s+', ' ', clean_state).title()
    
    # 3. Standardize ampersands by turning " & " into " And " for uniform evaluation
    clean_dist = clean_dist.replace(" & ", " And ")
    clean_state = clean_state.replace(" & ", " And ")

    # ---------------------------------------------------------
    # STATE-LEVEL CORRECTIONS (Evaluated using "And")
    # ---------------------------------------------------------
    state_fixes = {
        "Orissa": "Odisha",
        "Chhatisgarh": "Chhattisgarh",
        "Jammu And Kashmir": "Jammu And Kashmir",
        "Nct Of Delhi": "Delhi",
        "Andaman & Nicobar Islands": "Andaman and Nicobar",
        "Andaman And Nicobar Islands": "Andaman And Nicobar",
        "Andaman And Nicobar": "Andaman And Nicobar",
    }
    if clean_state in state_fixes:
        clean_state = state_fixes[clean_state]
        
    # ---------------------------------------------------------
    # DADRA & DAMAN SEPARATION LOGIC
    # ---------------------------------------------------------
    if clean_state == "Dadra And Nagar Haveli And Daman And Diu":
        if clean_dist == "Dadra And Nagar Haveli":
            clean_state = "Dadra And Nagar Haveli"
        elif clean_dist in ["Daman", "Diu"]:
            clean_state = "Daman And Diu"

    # ---------------------------------------------------------
    # DISTRICT-LEVEL CORRECTIONS 
    # ---------------------------------------------------------
    district_fixes = {
        "Central": "Central Delhi", "East": "East Delhi", "North": "North Delhi",
        "North East": "North East Delhi", "North West": "North West Delhi",
        "South": "South Delhi", "South East": "South East Delhi",
        "South West": "South West Delhi", "West": "West Delhi",
        "Bangalore": "Bengaluru", "Bangalore Rural": "Bengaluru Rural",
        "Belgaum": "Belagavi", "Bellary": "Ballari", "Chikmagalur": "Chikkamagaluru",
        "Gulbarga": "Kalaburagi", "Mysore": "Mysuru", "Shimoga": "Shivamogga",
        "Tumkur": "Tumakuru", "Bagalkot": "Bagalkote",
        "Darjiling": "Darjeeling", "Haora": "Howrah", "Hugli": "Hooghly",
        "Koch Bihar": "Cooch Behar", "Puruliya": "Purulia",
        "North Twenty Four Parganas": "North 24 Parganas",
        "South Twenty Four Parganas": "South 24 Parganas",
        "Paschim Barddhaman": "Paschim Bardhaman", "Purba Barddhaman": "Purba Bardhaman",
        "Paschim Medinipur": "Medinipur West", "Purba Medinipur": "Medinipur East",
        "Paschim Midnapore": "Medinipur West", "Purba Midnapore": "Medinipur East",
        "Allahabad": "Prayagraj", "Gurgaon": "Gurugram", "Mewat": "Nuh",
        "Jyotiba Phule Nagar": "Amroha", "Kanshiram Nagar": "Kasganj",
        "Mahamaya Nagar": "Hathras", "Sant Ravidas Nagar (Bhadohi)": "Bhadohi",
        "Charkhi Dadri": "Charki Dadri", "Garhwal": "Pauri Garhwal", "Hardwar": "Haridwar",
        "Bemetara": "Bametara", "Bemetra": "Bametara", "Janjgir Champa": "Janjgir Champa",
        "Kodagaon": "Kondagaon", "Dantewada": "Dakshin Bastar Dantewada", "Gariyaband": "Gariaband",
        "Jayashankar Bhupalpally": "Jayashankar", "Jayashankar Bhupalapally": "Jayashankar",
        "Komaram Bheem Asifabad": "Kumuram Bheem Asifabad", "Medchal Malkajgiri": "Medchal Malkajgiri",
        "Muktsar": "Sri Muktsar Sahib", "Sahibzada Ajit Singh Nagar": "S.A.S. Nagar",
        "Lahul and Spiti": "Lahul & Spiti",
        "Dibang Valley": "Upper Dibang Valley", "East Jantia Hills": "East Jaintia Hills",
        "Sepahijala": "Sipahijala", "Unakoti": "Unokoti",
        "Aravali": "Aravalli", "Chhota Udaipur": "Chota Udaipur", "Chhotaudepur": "Chota Udaipur",
        "Kaimur (Bhabua)": "Kaimur Bhabhua", "Kaimur": "Kaimur Bhabhua",
        "Pashchim Champaran": "West Champaran", "Purba Champaran": "East Champaran",
        "Khandwa (East Nimar)": "East Nimar", "Khandwa": "East Nimar",
        "Khargone (West Nimar)": "West Nimar", "Khargone": "West Nimar",
        "Nicobars": "Nicobars", "South Andaman": "South Andaman",
        "North  & Middle Andaman": "North And Middle Andaman",
        "North & Middle Andaman": "North And Middle Andaman",
        "Leh(Ladakh)": "Leh", "Jhunjhunun": "Jhunjhunu", "Sri Potti Sriramulu Nellore": "S.P.S. Nellore",
        "Aizawl": "Aizawal", "Janjgir - Champa": "Janjgir Champa",
        "Medchal-Malkajgiri": "Medchal Malkajgiri", "North District": "North  District",
    }
    
    if clean_dist in district_fixes:
        clean_dist = district_fixes[clean_dist]
        
    # 4. Strict Regional Conflict Rule
    if clean_dist == "Bijapur" and clean_state == "Karnataka":
        clean_dist = "Vijayapura"
    if clean_dist == "Chamarajanagar" and clean_state == "Karnataka":
        clean_dist = "Chamarajanagara"

    return f"{clean_dist}, {clean_state}"


def load_and_prep_csv(csv_file, states_json, districts_json, map_keys):
    print("Processing survey entries...")
    df = pd.read_csv(csv_file, low_memory=False)

    with open(states_json, 'r') as f:
        states_map = {int(k): str(v).strip().title() for k, v in json.load(f).items()}
    with open(districts_json, 'r') as f:
        dist_map = {int(k): str(v).strip().title() for k, v in json.load(f).items()}

    df['state_code'] = pd.to_numeric(df['state'], errors='coerce').fillna(-1).astype(int)
    df['district_code'] = pd.to_numeric(df['district'], errors='coerce').fillna(-1).astype(int)

    df['StateName'] = df['state_code'].map(states_map)
    df['DistName'] = df['district_code'].map(dist_map)
    df = df.dropna(subset=['StateName', 'DistName'])

    df['MatchKey'] = df.apply(lambda row: standardize_names(row['DistName'], row['StateName']), axis=1)

    print("\n" + "="*50)
    print(" 📊 DISTRICT MATCHING REPORT")
    print("="*50)

    csv_keys = set(df['MatchKey'].unique())
    matched_keys = csv_keys.intersection(map_keys)
    unmatched_csv = csv_keys - map_keys
    unmatched_map = map_keys - csv_keys

    print(f"Total distinct districts in your CSV: {len(csv_keys)}")
    print(f"Total distinct districts in the Map:  {len(map_keys)}")
    print(f"✅ Successfully Matched:             {len(matched_keys)}")

    if unmatched_csv:
        print(f"\n❌ WARNING: {len(unmatched_csv)} districts from your CSV did not find a map boundary.")
        print("You may need to add these to the 'standardize_names' function:")
        for k in sorted(unmatched_csv):
            print(f"   - {k}")
            
        print("\n💡 HINT: Here are the available MAP boundaries for those problem states:")
        problem_states = set([k.split(", ")[1] for k in unmatched_csv])
        for state in sorted(problem_states):
            print(f"\n  Available in Map for {state}:")
            available = [k for k in unmatched_map if k.endswith(state)]
            for a in sorted(available):
                print(f"     -> {a}")
    else:
        print("\n✅ SUCCESS: All CSV districts matched perfectly!")
    print("="*50 + "\n")

    # Drop unmatched rows
    df = df[df['MatchKey'].isin(matched_keys)]
    return df


def generate_plotly_map_html(map_data, geojson, states_geojson, states_gdf, color_col, hover_data_dict, labels_dict, map_title, color_scale="Reds", color_midpoint=None, custom_layout=None):
    """Generates the Plotly figure, attaches a Search UI, and an Overlay Card."""
    print("Building visual layers and Search UI...")

    fig = px.choropleth_map(
        map_data, 
        geojson=geojson,  
        locations='MatchKey',           
        featureidkey="properties.MatchKey", 
        color=color_col,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        hover_name="MatchKey",
        hover_data=hover_data_dict,
        labels=labels_dict,
        title=map_title,
        map_style="carto-positron", 
        center={"lat": 22.5, "lon": 79.0}, 
        zoom=4.1,
        opacity=0.75,
        height=800  
    )

    fig.add_trace(
        go.Choroplethmap(
            geojson=states_geojson,
            locations=states_gdf['st_nm_clean'],
            z=[1] * len(states_gdf), 
            featureidkey="properties.st_nm_clean",
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']], 
            showscale=False,        
            marker_opacity=1,
            marker_line_width=2.2,   
            marker_line_color='black', 
            hoverinfo='skip'        
        )
    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    if custom_layout:
        fig.update_layout(**custom_layout)
    
    # 1. Generate the raw Plotly DIV
    plotly_div = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plotly-map')
    
    # 2. Extract Data for JS
    options = "".join([f'<option value="{loc}">' for loc in sorted(map_data['MatchKey'].unique())])
    map_json = map_data.to_json(orient='records')
    labels_json = json.dumps(labels_dict)
    
    # 3. Build UI with floating result card
    search_ui = f"""
    <div style="margin-bottom: 15px; display: flex; gap: 10px; align-items: center; background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #ddd;">
        <span style="font-size: 20px;">🔍</span>
        <input type="text" id="mapSearch" list="locationOptions" 
               placeholder="Search for a district or state (e.g., 'Bengaluru' or 'Kerala')..." 
               style="flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 16px; outline: none;">
        <datalist id="locationOptions">
            {options}
        </datalist>
        <button onclick="triggerSearch()" 
                style="padding: 10px 20px; background: #2c3e50; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: 0.2s;">
            Look Up Data
        </button>
    </div>

    <div style="position: relative;">
        {plotly_div}
        
        <div id="custom-tooltip" style="display: none; position: absolute; top: 20px; right: 20px; width: 320px; background: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,0.3); border-left: 5px solid #e74c3c; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; z-index: 1000;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ddd; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 id="tooltip-title" style="margin: 0; color: #2c3e50; font-size: 16px;"></h3>
                <span onclick="document.getElementById('custom-tooltip').style.display='none'" style="cursor: pointer; font-size: 20px; color: #999; line-height: 1;">&times;</span>
            </div>
            <div id="tooltip-body" style="font-size: 14px; color: #333;"></div>
        </div>
    </div>

    <script>
        const sourceData = {map_json};
        const labelsDict = {labels_json};

        function triggerSearch() {{
            const query = document.getElementById('mapSearch').value.toLowerCase().trim();
            if (!query) return;

            let foundRow = sourceData.find(row => row.MatchKey.toLowerCase() === query);
            if (!foundRow) {{
                foundRow = sourceData.find(row => row.MatchKey.toLowerCase().includes(query));
            }}

            const tooltip = document.getElementById('custom-tooltip');
            
            if (foundRow) {{
                document.getElementById('tooltip-title').innerText = "📍 " + foundRow.MatchKey;
                
                let bodyHtml = '<table style="width:100%; border-collapse: collapse;">';
                for (const [key, val] of Object.entries(foundRow)) {{
                    if (key === 'MatchKey') continue;
                    
                    // Use clean dictionary label if exists, else format the raw key
                    let cleanLabel = labelsDict[key] || key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                    
                    // Format numeric values
                    let cleanVal = val;
                    if (typeof val === 'number') {{
                        cleanVal = val % 1 === 0 ? val : val.toFixed(4);
                    }}
                    
                    bodyHtml += `<tr>
                        <td style="padding: 6px 0; font-weight: 600; color: #555; border-bottom: 1px solid #eee;">${{cleanLabel}}:</td>
                        <td style="padding: 6px 0; text-align: right; border-bottom: 1px solid #eee;">${{cleanVal}}</td>
                    </tr>`;
                }}
                bodyHtml += '</table>';
                
                document.getElementById('tooltip-body').innerHTML = bodyHtml;
                tooltip.style.display = 'block';
                
                // Slight pop animation to draw the eye
                tooltip.animate([
                    {{ transform: 'scale(0.95)', opacity: 0 }},
                    {{ transform: 'scale(1)', opacity: 1 }}
                ], {{ duration: 200, easing: 'ease-out' }});
                
            }} else {{
                alert("Location '" + query + "' not found in this dataset.");
                tooltip.style.display = 'none';
            }}
        }}

        // Trigger on Enter key
        document.addEventListener('DOMContentLoaded', function() {{
            const searchInput = document.getElementById('mapSearch');
            if(searchInput) {{
                searchInput.addEventListener('keypress', function (e) {{
                    if (e.key === 'Enter') triggerSearch();
                }});
            }}
        }});
    </script>
    """
    
    return search_ui