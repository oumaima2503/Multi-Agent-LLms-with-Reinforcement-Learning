"""
Génère un Jupyter Notebook récapitulatif pour un agent à partir des CSV dans outputs/{agent}/
Usage:
    pip install nbformat pandas plotly
    python export_report_notebook.py --agent orchestrator
"""
import argparse
import os
try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
except Exception as e:
    raise RuntimeError("Installez nbformat: pip install nbformat") from e

def generate_notebook_for_agent(agent, outputs_root="outputs"):
    csv_path = os.path.join(outputs_root, agent, f"{agent}_comparison_summary.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    nb = new_notebook()
    nb.cells = []
    nb.cells.append(new_markdown_cell(f"# Rapport comparatif - {agent}\nGénéré automatiquement à partir de `{csv_path}`"))
    code1 = f"""import pandas as pd
import plotly.express as px
df = pd.read_csv(r"{csv_path}", index_col=0)
df"""
    nb.cells.append(new_code_cell(code1))
    code2 = """fig1 = px.bar(df.reset_index(), x='checkpoint', y='reward', title='Reward comparatif')
fig1.show()"""
    nb.cells.append(new_code_cell(code2))
    code3 = """fig2 = px.bar(df.reset_index(), x='checkpoint', y='time_s', title='Temps de réponse (s)')
fig2.show()"""
    nb.cells.append(new_code_cell(code3))
    code4 = """if 'delegated_agent' in df.columns:
    counts = df['delegated_agent'].fillna('None').value_counts().reset_index()
    counts.columns = ['delegated_agent','count']
    fig3 = px.bar(counts, x='delegated_agent', y='count', title='Distribution des délégations')
    fig3.show()
else:
    print('Champ delegated_agent absent')"""
    nb.cells.append(new_code_cell(code4))
    out_dir = os.path.join(outputs_root, agent)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{agent}_report.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--outputs", type=str, default="outputs")
    args = parser.parse_args()
    try:
        path = generate_notebook_for_agent(args.agent, outputs_root=args.outputs)
        print(f"Notebook généré: {path}")
    except Exception as e:
        print(f"Erreur: {e}")