
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
import plotly.express as px

# ACS data
# Note - variables can be found here https://api.census.gov/data/2023/acs/acs5/variables.html
# Get your own API key here: https://api.census.gov/data/key_signup.html

API_KEY = 'd0925e3f914c6d30be084874b1a6b018f3559860' # Your own census API key

# Years we'll run for
years = [2018, 2019, 2020, 2021, 2022, 2023]

# Variables we want - borrowing some from Fall 2024's script for this
variables = [
    "B19013_001E",  # Median Household Income
    "B25064_001E",  # Median Gross Rent
    "B25070_007E",  # Rent-burdened households (30-34.9% of income on rent)
    "B25070_008E",  # Rent-burdened households (35-39.9% of income on rent)
    "B25070_009E",  # Rent-burdened households (40-49.9% of income on rent)
    "B25070_010E",  # Rent-burdened households (50% or more of income on rent)
    "B25070_001E",   # Total Renter-occupied Units
    "B25077_001E" # Median home value (could also look at distribution with B25075 table)

]

# Pull ACS via Census API - borrowing some from Fall 2024's script for this

def fetch_census_data(year, place_type, fips, state_fips=None):
  base_url = f"https://api.census.gov/data/{year}/acs/acs5"

  if place_type == "place":
    if not state_fips:
      raise ValueError("State FIPS is required for place")
    params = {
        "get": ",".join(variables),
        "for": f"place:{fips}",
        "in": f"state:{state_fips}",
        "key": API_KEY
        }
  elif place_type == "state":
    params = {
        "get": ",".join(variables),
        "for": f"state:{fips}",
        "key": API_KEY
    }
  elif place_type == "msa":
    params = {
        "get": ",".join(variables),
        "for": f"metropolitan statistical area/micropolitan statistical area:{fips}",
        "key": API_KEY
    }
  else:
    raise ValueError("Choose from 'place', 'state', or 'msa'.")

  response = requests.get(base_url, params=params)
  if response.status_code != 200:
    print(f"Error fetching data for {year}: {response.status_code}, {response.text}")
    return None
  return response.json()

# Needed for application; saves data and doesn't refresh every single time (would take forever to load without this)
@st.cache_data

# Process and clean data
def process_data(year, place_type, fips, state_fips=None):
  records = []
  for year in years:
    data = fetch_census_data(year, place_type, fips, state_fips)
    if data:
      header, values = data[0], data[1]
      record = dict(zip(header, values))
      record["year"] = year
      records.append(record)
  df = pd.DataFrame(records)
  # Rename columns for clarity
  df.rename(columns={
      "B19013_001E": "median_income",
      "B25064_001E": "median_rent",
      "B25070_007E": "rent_burden_30_34",
      "B25070_008E": "rent_burden_35_39",
      "B25070_009E": "rent_burden_40_49",
      "B25070_010E": "rent_burden_50_plus",
      "B25070_001E": "total_renters",
      "B25077_001E": "median_hval"
  }, inplace=True)
  # Convert columns to numeric
  numeric_columns = ["median_income", "median_rent", "rent_burden_30_34", "rent_burden_35_39", "rent_burden_40_49", "rent_burden_50_plus", "total_renters", "median_hval"]
  for col in numeric_columns:
      if col in df.columns:
          df[col] = pd.to_numeric(df[col], errors="coerce")
  # Calculate total rent-burdened households
  df["rent_burdened_total"] = df["rent_burden_30_34"] + df["rent_burden_35_39"] + df["rent_burden_40_49"] + df["rent_burden_50_plus"]
  # Calculate rent-burdened percentage
  df["rent_burdened_pct"] = (df["rent_burdened_total"] / df["total_renters"]) * 100
  return df

# Run for Bloomington city
# Place 05860 and state 18
acs_bloom_city = process_data(years, "place", "05860", state_fips="18")
# Run for bloom msa too
acs_bloom_msa = process_data(years, "msa", "14020")
# Run again for IN overall
acs_in = process_data(years, "state", "18")

# Save all 3 as csv files
acs_bloom_city.to_csv('acs_bloom_city.csv',index=False)
acs_bloom_msa.to_csv('acs_bloom_msa.csv',index=False)
acs_in.to_csv('acs_in.csv',index=False)

# Load the three data files
acs_bloom_city = pd.read_csv('acs_bloom_city.csv')
acs_bloom_msa = pd.read_csv('acs_bloom_msa.csv')
acs_in = pd.read_csv('acs_in.csv')

# BLS data - OEWS data for Bloomington MSA 2023 - contains wages, employment by job type
# Sourced from https://www.bls.gov/oes/2023/may/oes_14020.htm
# Load this file too (can be found in basecamp and the github repo)
bls_bloom = pd.read_csv('bls_bloom.csv')

# Add dataset labels
acs_bloom_city['dataset'] = 'Bloomington City'
acs_bloom_msa['dataset'] = 'Bloomington Metro'
acs_in['dataset'] = 'Indiana'

# Remove rows where the last 4 digits of OCC_CODE are '0000'
occ_types = bls_bloom[~bls_bloom["OCC_CODE"].str.endswith("0000")].copy()
occ_types["A_MEDIAN"] = pd.to_numeric(occ_types["A_MEDIAN"], errors="coerce")

# Calculate ren burden for each occupation
median_rent_2023 = acs_bloom_msa[acs_bloom_msa['year'] == 2023]['median_rent'].iloc[0]
annual_rent = median_rent_2023 * 12
annual_income = occ_types['A_MEDIAN']

occ_types['percent_rent_burdened'] = (annual_rent / annual_income) * 100
occ_types['percent_rent_burdened'] = occ_types['percent_rent_burdened'].where(
    occ_types['percent_rent_burdened'] > 30, 0
)
rent_burdened_occupations = occ_types[occ_types['percent_rent_burdened'] > 0]

# Combine into a single DataFrame
combined_df = pd.concat([acs_bloom_city, acs_bloom_msa, acs_in], ignore_index=True)
combined_df['year'] = combined_df['year'].astype(int)

# Color map for areas (red, blue, purple)
color_map = {
    'Indiana': '#8B0000',
    'Bloomington City': '#003366',
    'Bloomington Metro': '#800080'
}

# Streamlit app
def main():
    st.title('City of Bloomington Housing Trends')

    # Refreshes data (likely only used once or twice a year)
    if st.button("Refresh Data"):
      st.cache_data.clear()
      st.rerun()

    # Tabs on left side of application
    tab = st.sidebar.radio("Select View", ['Housing', 'Rent-Burdened Occupations'])
    # Graphs in Housing section
    if tab == 'Housing':
        metric = st.sidebar.selectbox("Select Visual", ['Median Rent', 'Median Home Value'])
        st.subheader(f"{metric} Over Time by Location")

        st.sidebar.header("Filters")
        min_year = combined_df['year'].min()
        max_year = combined_df['year'].max()
        year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (2018, 2023))

        metric_column = 'median_rent' if metric == 'Median Rent' else 'median_hval'
        y_label = 'Median Rent ($)' if metric == 'Median Rent' else 'Median Home Value ($)'

        selected_datasets = st.sidebar.multiselect(
            "Select Area",
            options=combined_df['dataset'].unique(),
            default=combined_df['dataset'].unique()
        )

        filtered_df = combined_df[
            (combined_df['year'] >= year_range[0]) &
            (combined_df['year'] <= year_range[1]) &
            (combined_df['dataset'].isin(selected_datasets))
        ]

        tick_vals = list(range(year_range[0], year_range[1] + 1))

        fig = px.line(
            filtered_df,
            x='year',
            y=metric_column,
            color='dataset',
            markers=True,
            color_discrete_map=color_map,
            labels={
                'year': 'Year',
                metric_column: y_label,
                'dataset': 'Region'
            },
            width=1000,
            height=600
        )
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=tick_vals))
        st.plotly_chart(fig, use_container_width=True)
    # Graph in Rent-Burdened Occupations section
    elif tab == 'Rent-Burdened Occupations':
        st.subheader("Which workers are rent-burdened in Bloomington?")

        top_15_by_employees = rent_burdened_occupations.nlargest(15, "TOT_EMP")[[
            "OCC_TITLE", "TOT_EMP", "A_MEDIAN", "percent_rent_burdened"
        ]]
        top_15_by_employees = top_15_by_employees.sort_values("TOT_EMP", ascending=True)

        fig = px.bar(
            top_15_by_employees,
            x="TOT_EMP",
            y="OCC_TITLE",
            orientation="h",
            color="percent_rent_burdened",
            color_continuous_scale="Blues",
            labels={
                "OCC_TITLE": "Occupation",
                "TOT_EMP": "Number of Employees",
                "percent_rent_burdened": "Percent rent-burdened"
            },
            title="Top 15 Rent-Burdened Occupations by Employment",
        )

        fig.update_traces(marker_line_color="gray", marker_line_width=1)

        fig.update_layout(
            height=800,
            xaxis_title="Number of Employees",
            yaxis_title="",
            coloraxis_colorbar=dict(title="Percent rent-burdened"),
            template="plotly_white",
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

# Warning message to let city know this is a DRAFT version and is NOT finished in its current state.
st.warning("This is a DRAFT version of the app. This is NOT the final version.", icon="⚠️")

# Runs app.py
if __name__ == "__main__":
    main()
