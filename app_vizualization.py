"""
Interactive 3D t-SNE Visualization for Musical Acoustical Similarity
--------------------------------------------------------------------

This application computes a 3D t-SNE embedding of a multidimensional acoustic
feature set (timbre, pitch-class profile, key, mode, tempo, time signature,
loudness) and provides an interactive Dash interface for exploring musical
similarity. Users can filter by gender, country, genre, and year range, and
dynamically recolor the projection by genre, gender, or continent. A right-hand
panel displays metadata for the selected track and its five nearest neighbors in
t-SNE space.
"""

import numpy as np
import pandas as pd
import pycountry
import pycountry_convert as pc
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px

import dash

from dash import Dash, dcc, html, Input, Output


# ================================================================
# 0. Dash App Initialization with Modern Font
# ================================================================

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600;700&display=swap",
        "rel": "stylesheet",
    }
]

app = Dash(__name__, external_stylesheets=external_stylesheets)


# ================================================================
# 1. Load and Validate Dataset
# ================================================================

df = pd.read_csv("final.csv")

# Base metadata required
required_base_cols = ["genre_principal", "artist_name", "title", "year", "gender", "country"]

# Audio feature set
timbre_cols = [f"timbre_{i}_mean" for i in range(12)]
pitch_cols = [f"pitch_{i}_mean" for i in range(12)]
additional_audio_cols = ["key", "mode", "tempo", "time_signature", "loudness"]

feature_cols = timbre_cols + pitch_cols + additional_audio_cols
required_cols = required_base_cols + feature_cols

# Column presence check (fails fast if CSV is incomplete)
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from final.csv")


# ================================================================
# 2. Data Cleaning
# ================================================================

df["year_int"] = df["year"].astype("Int64")
df["year_clean"] = df["year_int"].astype(str)


def clean_gender(g):
    """Normalize gender values into Male / Female / Unknown."""
    if isinstance(g, str):
        gl = g.lower()
        if "female" in gl:
            return "Female"
        if "male" in gl:
            return "Male"
    return "Unknown"


df["gender_clean"] = df["gender"].apply(clean_gender)


def convert_country(c):
    """Convert ISO alpha-2/3 country codes to full names; otherwise title-case string."""
    if not isinstance(c, str) or not c.strip():
        return "Unknown"
    code = c.strip().upper()

    ctry = pycountry.countries.get(alpha_2=code) or pycountry.countries.get(alpha_3=code)
    return ctry.name if ctry else c.title()


df["country_clean"] = df["country"].apply(convert_country)


def country_to_continent(name):
    """Map country name to continent name using pycountry_convert."""
    try:
        ctry = pycountry.countries.get(name=name)
        if not ctry:
            return "Unknown"
        cont = pc.country_alpha2_to_continent_code(ctry.alpha_2)
        return pc.convert_continent_code_to_continent_name(cont)
    except Exception:
        return "Unknown"


df["continent_clean"] = df["country_clean"].apply(country_to_continent)


# ================================================================
# 3. Compute t-SNE Embedding (once at launch)
# ================================================================

# Keep only rows with a declared genre
all_genres = sorted(df["genre_principal"].dropna().unique())
subset = df[df["genre_principal"].isin(all_genres)].copy()

# Drop rows with missing acoustic features (timbre + pitch)
subset = subset.dropna(subset=timbre_cols + pitch_cols).copy()

# Build feature matrix for the acoustic embedding (29 dimensions)
X = subset[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = TSNE(
    n_components=3,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    metric="euclidean",
    random_state=42,
)

X_tsne = tsne.fit_transform(X_scaled)

# Build visualization dataframe
vis_df = subset.copy()
vis_df["tsne_1"] = X_tsne[:, 0]
vis_df["tsne_2"] = X_tsne[:, 1]
vis_df["tsne_3"] = X_tsne[:, 2]

vis_df = vis_df.reset_index(drop=False).rename(columns={"index": "row_id"})
vis_df = vis_df[
    [
        "row_id",
        "tsne_1",
        "tsne_2",
        "tsne_3",
        "title",
        "artist_name",
        "year_int",
        "year_clean",
        "gender_clean",
        "country_clean",
        "continent_clean",
        "genre_principal",
    ]
]


# ================================================================
# 4. User Interface
# ================================================================

available_genders = sorted(vis_df["gender_clean"].unique())
available_countries = sorted(vis_df["country_clean"].unique())
available_genres = sorted(vis_df["genre_principal"].unique())
available_continents = sorted(vis_df["continent_clean"].unique())

year_min = int(vis_df["year_int"].min())
year_max = int(vis_df["year_int"].max())

unique_years = sorted(vis_df["year_int"].dropna().unique())
marks = (
    {int(y): str(int(y)) for y in unique_years}
    if len(unique_years) <= 20
    else {y: str(y) for y in range(year_min, year_max + 1, max((year_max - year_min) // 10, 1))}
)

app.layout = html.Div(
    style={"backgroundColor": "#1e1e1e", "color": "white", "fontFamily": "'Inter Tight', sans-serif"},
    children=[
        

        html.H1(
            "Navigating the Musical Landscape: 3D Similarity Map",
            style={"textAlign": "center", "fontWeight": "600", "marginBottom": "20px"},
        ),

        # Filters (Color-by placed on the left)
        html.Div(
            style={
                "display": "flex",
                "gap": "40px",
                "marginBottom": "20px",
                "justifyContent": "center",
            },
            children=[
                # 1. Color points by (moved to the left)
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Color points by"),
                        dcc.Dropdown(
                            id="color-by",
                            className="dark-dropdown",
                            options=[
                                {"label": "Genre", "value": "genre_principal"},
                                {"label": "Gender", "value": "gender_clean"},
                                {"label": "Continent", "value": "continent_clean"},
                            ],
                            value="genre_principal",
                            clearable=False,
                        ),
                    ],
                ),
                # 2. Gender
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Gender"),
                        dcc.Checklist(
                            id="gender-filter",
                            options=[{"label": g, "value": g} for g in available_genders],
                            value=available_genders,
                            inline=True,
                            style={"color": "white"},
                        ),
                    ],
                ),
                # 3. Country
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Country"),
                        dcc.Dropdown(
                            id="country-filter",
                            className="dark-dropdown",
                            options=[{"label": c, "value": c} for c in available_countries],
                            value=available_countries,
                            multi=True,
                        ),
                    ],
                ),
                # 4. Genre
                html.Div(
                    style={"minWidth": "220px"},
                    children=[
                        html.Label("Genre"),
                        dcc.Dropdown(
                            id="genre-filter",
                            className="dark-dropdown",
                            options=[{"label": g, "value": g} for g in available_genres],
                            value=available_genres,
                            multi=True,
                        ),
                    ],
                ),
            ],
        ),

        # Year slider
        html.Div(
            children=[
                html.Label("Year range"),
                dcc.RangeSlider(
                    id="year-range",
                    min=year_min,
                    max=year_max,
                    value=[year_min, year_max],
                    marks=marks,
                ),
            ]
        ),

        # Graph + Panel
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                html.Div(
                    style={"flex": "3"},
                    children=[dcc.Graph(id="tsne-3d-graph", style={"height": "750px"})],
                ),
                html.Div(
                    id="right-panel",
                    style={
                        "flex": "2",
                        "backgroundColor": "#2a2a2a",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "overflowY": "auto",
                        "height": "750px",
                    },
                    children=[
                        html.H3("Selected track"),
                        html.Div(id="selected-track"),
                        html.H3("5 nearest neighbors", style={"marginTop": "20px"}),
                        html.Div(id="neighbors-table"),
                    ],
                ),
            ],
        ),
    ],
)


# ================================================================
# 5. Callback
# ================================================================

@app.callback(
    Output("tsne-3d-graph", "figure"),
    Output("selected-track", "children"),
    Output("neighbors-table", "children"),
    [
        Input("gender-filter", "value"),
        Input("country-filter", "value"),
        Input("genre-filter", "value"),
        Input("year-range", "value"),
        Input("color-by", "value"),
        Input("tsne-3d-graph", "clickData"),
    ],
)
def update_tsne(selected_genders, selected_countries, selected_genres, year_range, color_by, clickData):
    """Update the t-SNE projection and metadata according to filter selections."""

    dff = vis_df.copy()

    # Filters
    if selected_genders:
        dff = dff[dff["gender_clean"].isin(selected_genders)]
    if selected_countries:
        dff = dff[dff["country_clean"].isin(selected_countries)]
    if selected_genres:
        dff = dff[dff["genre_principal"].isin(selected_genres)]
    if year_range:
        dff = dff[(dff["year_int"] >= year_range[0]) & (dff["year_int"] <= year_range[1])]

    # Remove Unknown when coloring by gender
    if color_by == "gender_clean":
        dff = dff[dff["gender_clean"] != "Unknown"]

    if dff.empty:
        return px.scatter_3d(), html.Div("No track selected."), html.Div("No data.")

    # Plot
    fig = px.scatter_3d(
        dff,
        x="tsne_1",
        y="tsne_2",
        z="tsne_3",
        color=color_by,
        opacity=0.95,
        labels={
            "genre_principal": "Genre",
            "gender_clean": "Gender",
            "continent_clean": "Continent",
        }
    )

    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        scene=dict(
            bgcolor="#1e1e1e",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )

    # Hover
    for trace in fig.data:
        mask = dff[color_by] == trace.name
        trace.customdata = dff.loc[
            mask,
            [
                "row_id",
                "title",
                "artist_name",
                "year_clean",
                "gender_clean",
                "country_clean",
                "continent_clean",
                "genre_principal",
            ],
        ].values

        trace.hovertemplate = (
            "<b>%{customdata[1]}</b><br>"
            "Artist: %{customdata[2]}<br>"
            "Year: %{customdata[3]}<br>"
            "Gender: %{customdata[4]}<br>"
            "Country: %{customdata[5]}<br>"
            "Continent: %{customdata[6]}<br>"
            "Genre: %{customdata[7]}<br>"
            "<extra></extra>"
        )

    # Default panels
    selected_track = html.Div("Click a point.")
    neighbors_table = html.Div("Click a point.")

    # If a point is selected
    if clickData and "points" in clickData:
        custom = clickData["points"][0].get("customdata")
        if custom is not None:
            clicked_id = int(custom[0])

            if clicked_id in dff["row_id"].values:
                selected_row = dff[dff["row_id"] == clicked_id].iloc[0]

                selected_track = html.Div(
                    [
                        html.H4(selected_row["title"]),
                        html.P(f"Artist: {selected_row['artist_name']}"),
                        html.P(f"Year: {selected_row['year_clean']}"),
                        html.P(f"Gender: {selected_row['gender_clean']}"),
                        html.P(f"Country: {selected_row['country_clean']}"),
                        html.P(f"Continent: {selected_row['continent_clean']}"),
                        html.P(f"Genre: {selected_row['genre_principal']}"),
                    ]
                )

                coords = dff[["tsne_1", "tsne_2", "tsne_3"]].values
                target = selected_row[["tsne_1", "tsne_2", "tsne_3"]].values.astype(float)

                dist = np.linalg.norm(coords - target, axis=1)

                temp = dff.copy()
                temp["distance"] = dist

                neighbors = temp[temp["row_id"] != clicked_id].sort_values("distance").head(5)

                rows = [
                    html.Tr(
                        [
                            html.Td(r["title"]),
                            html.Td(r["artist_name"]),
                            html.Td(str(r["year_clean"])),
                            html.Td(r["country_clean"]),
                            html.Td(r["genre_principal"]),
                            html.Td(f"{r['distance']:.3f}"),
                        ]
                    )
                    for _, r in neighbors.iterrows()
                ]

                neighbors_table = html.Table(
                    [
                        html.Tr(
                            [
                                html.Th("Title"),
                                html.Th("Artist"),
                                html.Th("Year"),
                                html.Th("Country"),
                                html.Th("Genre"),
                                html.Th("Dist"),
                            ]
                        )
                    ]
                    + rows,
                    style={"width": "100%"},
                )

    return fig, selected_track, neighbors_table


# ================================================================
# 6. Run Application
# ================================================================

if __name__ == "__main__":
    print("Dash app starting on http://127.0.0.1:8051/")
    print("If you see this message and no traceback, the server is running.")
    app.run(debug=True, port=8051)
