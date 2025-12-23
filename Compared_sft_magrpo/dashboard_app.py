"""
Dash dashboard pour visualiser les résultats de compare_sft_magrpo.py
Usage:
    pip install dash pandas plotly
    python dashboard_app.py --port 8050
"""
import os
import argparse
try:
    import pandas as pd
    import plotly.express as px
    from dash import Dash, dcc, html, Output, Input, dash_table
except Exception as e:
    raise RuntimeError("Installez dash, pandas, plotly: pip install dash pandas plotly") from e

def find_agents(outputs_root="outputs"):
    agents = []
    if not os.path.exists(outputs_root):
        return agents
    for name in os.listdir(outputs_root):
        p = os.path.join(outputs_root, name)
        if os.path.isdir(p):
            agents.append(name)
    return sorted(agents)

def load_agent_df(agent, outputs_root="outputs"):
    csv_path = os.path.join(outputs_root, agent, f"{agent}_comparison_summary.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, index_col=0)
    # normalize delegated_agent column if exists
    if "delegated_agent" in df.columns:
        df["delegated_agent"] = df["delegated_agent"].fillna("None").astype(str)
    return df

def create_app(outputs_root="outputs"):
    app = Dash(__name__)
    agents = find_agents(outputs_root)
    app.layout = html.Div([
        html.H3("MAGRPO vs SFT - Dashboard"),
        html.Div([
            html.Label("Agent:"),
            dcc.Dropdown(id="agent-select", options=[{"label":a,"value":a} for a in agents], value=agents[0] if agents else None)
        ], style={"width":"300px"}),
        html.Br(),
        html.Div(id="summary-area"),
        html.Br(),
        html.Div([
            dcc.Tabs(id="tabs", value="tab-1", children=[
                dcc.Tab(label="Tableau", value="tab-1"),
                dcc.Tab(label="Reward / Temps / Structure", value="tab-2"),
                dcc.Tab(label="Délégations", value="tab-3")
            ])
        ]),
        html.Div(id="tab-content")
    ], style={"margin":"20px"})
    
    @app.callback(Output("summary-area","children"), Input("agent-select","value"))
    def update_summary(agent):
        if not agent:
            return "Aucun agent trouvé dans outputs/"
        df = load_agent_df(agent, outputs_root)
        if df is None:
            return f"Aucun CSV pour {agent} (attendez que compare_sft_magrpo.py ait généré outputs/{agent})"
        rows = [
            html.P(f"Résultats: {len(df)} checkpoints"),
            html.P(f"Reward moyen: {df['reward'].mean():.2f}"),
            html.P(f"Temps moyen (s): {df['time_s'].mean():.2f}"),
            html.P(f"Nombre moyen de clés: {df['num_keys'].mean():.2f}")
        ]
        return rows

    @app.callback(Output("tab-content","children"), Input("agent-select","value"), Input("tabs","value"))
    def render_tab(agent, tab):
        if not agent:
            return "Sélectionnez un agent."
        df = load_agent_df(agent, outputs_root)
        if df is None:
            return f"Aucun CSV pour {agent}"
        if tab == "tab-1":
            return dash_table.DataTable(
                data=df.reset_index().to_dict("records"),
                columns=[{"name":c,"id":c} for c in df.reset_index().columns],
                page_size=10,
                style_table={"overflowX":"auto"},
            )
        elif tab == "tab-2":
            fig_reward = px.bar(df.reset_index(), x="checkpoint", y="reward", title=f"{agent} - Reward", labels={"reward":"Reward"})
            fig_time = px.bar(df.reset_index(), x="checkpoint", y="time_s", title=f"{agent} - Temps (s)", labels={"time_s":"Temps (s)"})
            fig_struct = px.bar(df.reset_index(), x="checkpoint", y="num_keys", title=f"{agent} - Nombre de clés", labels={"num_keys":"Nombre de clés"})
            return html.Div([
                dcc.Graph(figure=fig_reward),
                dcc.Graph(figure=fig_time),
                dcc.Graph(figure=fig_struct)
            ])
        else:
            if "delegated_agent" not in df.columns:
                return "Champ delegated_agent absent dans le CSV."
            counts = df["delegated_agent"].fillna("None").value_counts().reset_index()
            counts.columns = ["delegated_agent","count"]
            fig = px.bar(counts, x="delegated_agent", y="count", title=f"{agent} - Distribution des délégations")
            return dcc.Graph(figure=fig)

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--outputs", type=str, default="outputs")
    args = parser.parse_args()
    app = create_app(outputs_root=args.outputs)
    app.run_server(host=args.host, port=args.port, debug=False)