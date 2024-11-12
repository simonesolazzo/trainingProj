from flask import Flask, render_template, request
from datetime import timedelta
import pandas as pd
import re
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, CustomJSTickFormatter
from sklearn.neighbors import KernelDensity
from bokeh.palettes import Dark2_4 as colors # Palette con 4 colori
import numpy as np

app = Flask(__name__)
app.secret_key = 'key'
log = app.logger
ITEMS_PER_PAGE = 25

def format_duration(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days} giorni, {hours} ore, {minutes} minuti, {seconds} secondi"


def load_data(filepath='data/datamin.csv'):
    df = pd.read_csv(filepath)
    completed_process_logs = df[df['logType_id'] == 11]
    processes_info = [
        (
            re.search(r'The process with id: (\d+) and tenant domain: \w+-\d+ has been completed', row['message']).group(1), # Process ID
            pd.to_datetime(row['date']),                                    # Dataora fine processo
            row['username']                                                 # Username
        )
        for _, row in completed_process_logs.iterrows()
    ]
    return df, processes_info


def get_completed_processes():
    df, processes_info = load_data()
    completed_processes = []
    
    for process_id, end_date, username in processes_info:
        started_process_logs = df[
            (df['logType_id'] == 9) &
            (df['message'].str.contains(f'workflow-id: {process_id}'))
        ]
        # Recupero il tipo di firma
        signature_logs = df[(df['logType_id'] == 8) & (df['message'].str.contains(f'Process id: {process_id}'))]
        signature_type = None
        if not signature_logs.empty:
            match = re.search(r"sign  by (\w+)", signature_logs.iloc[0]['message'])
            signature_type = match.group(1) if match else None  
        for _, row in started_process_logs.iterrows():
            start_date = pd.to_datetime(row['date'])
            duration_hours = (end_date - start_date).total_seconds() / 3600            
            completed_processes.append({
                'id': process_id,
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration_hours,
                'username': username,
                'signature': signature_type  # Aggiunge il tipo di firma
            }) 
    return completed_processes


def get_bins(processes, signature_type=None):
    bins = []
    types = ["firmaSemplice", "firmaAvanzata", "firmaAvanzataOTP", "firmaQualificata"]
    for i, sig_type in enumerate(types):
        label = sig_type
        times = [p['duration'] for p in processes if p['signature'] == sig_type]
        if signature_type is None or signature_type == sig_type:
            bins.append((label, times, colors[i]))
    return bins


def create_density_plot(processes, signature_type=None):
    # Determino l'unità dell'asse X (ore o giorni)
    execution_times = [p['duration'] for p in processes]
    max_time = max(execution_times) if execution_times else 0
    is_in_hours = max_time <= 24
    x_label = "Durata (hh:mm)" if is_in_hours else "Durata (giorni)"
    if not is_in_hours:
        execution_times = [t / 24 for t in execution_times]  # Converti tutte le durate in giorni

    bins = get_bins(processes, signature_type)

    plot = figure(
        x_axis_label=x_label,
        y_axis_label="Numero di processi",
        height=400,
        sizing_mode="stretch_width"
        # ,
        # background_fill_color="#1c1e21",
        # border_fill_color="#1c1e21"
    )
    # plot.xaxis.axis_label_text_color = "white"
    # plot.yaxis.axis_label_text_color = "white"
    # plot.xaxis.major_label_text_color = "white"
    # plot.yaxis.major_label_text_color = "white"
    # plot.outline_line_color = "white"
    # plot.axis.axis_line_color = "white"
    # plot.axis.major_tick_line_color = "white"
    # plot.axis.minor_tick_line_color = "white"
    # plot.grid.grid_line_color = "#444444"

    # Se siamo in ore, applica il formato hh:mm sull'asse x
    if is_in_hours:
        plot.xaxis.formatter = CustomJSTickFormatter(code="""
            const hours = Math.floor(tick);
            const minutes = Math.round((tick % 1) * 60);
            return `${hours}:${minutes < 10 ? '0' : ''}${minutes}`;
        """)

    hover = HoverTool(tooltips=[
        ("Tempo di esecuzione", "@formatted_duration"),
        ("Numero di processi", "@process_count_rounded"),
        ("Totale processi nell'area", "@total_processes")
    ])
    plot.add_tools(hover)
    plot.toolbar.logo = None
    plot.toolbar_location = None

    legend_items = []
    max_process_count = 0

    # Creo una densità per ogni tipo di firma e disegno come area
    for i, (label, times, color) in enumerate(bins):
        if not times:
            log.warning(f"Nessun dato disponibile per la firma: {label}")
            continue  # Salta se non ci sono dati

        if not is_in_hours:
            times = [t / 24 for t in times]
        X = np.array(times)[:, np.newaxis]
        kde = KernelDensity(bandwidth=1, kernel='gaussian')
        kde.fit(X)

        x_vals = np.linspace(0, max(execution_times), 1000)[:, np.newaxis]
        log_density = kde.score_samples(x_vals)
        y_vals = np.exp(log_density)

        process_counts = y_vals * len(times)
        # process_counts = np.ceil(y_vals * len(times))

        max_process_count = max(max_process_count, process_counts.max())

        source_data = {
            'x': x_vals.flatten(),
            'y': process_counts,
            'process_count_rounded': np.round(process_counts).astype(int),
            'total_processes': [len(times)] * len(x_vals),
            'formatted_duration': [
                format_duration(t * 3600) if is_in_hours else format_duration(t * 86400)
                for t in x_vals.flatten()
            ],
            'id_area': [i + 1] * len(x_vals)
        }
        source = ColumnDataSource(data=source_data)

        plot.varea(x='x', y1=0, y2='y', color=color, alpha=0.6, source=source)
        legend_items.append((label, color, i + 1))

    plot.y_range.end = max_process_count * 1.1

    return plot, legend_items


def filter_sort_paginate(processes, search_query='', sort_by='start_date', order='asc', page=1, items_per_page=ITEMS_PER_PAGE):
    filtered_processes = [
        {**p, 'formatted_duration': format_duration(((p['duration']*3600)))}
        for p in processes
        if search_query.lower() in p['username'].lower()
    ]

    reverse = (order == 'desc')
    filtered_processes.sort(key=lambda x: x[sort_by], reverse=reverse)

    total_items = len(filtered_processes)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    
    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page
    paginated_processes = filtered_processes[start_index:end_index]

    has_next_page = page < total_pages

    return paginated_processes, has_next_page, total_pages


@app.route('/')
def index():
    completed_processes = get_completed_processes()
    plot, legend_items = create_density_plot(completed_processes)
    script, div = components(plot)

    return render_template(
        'index.html',
        script=script,
        div=div,
        legend_items=legend_items
    )

@app.route('/details/<id_area>')
def area_detail(id_area):
    signature_types = ["firmaSemplice", "firmaAvanzata", "firmaAvanzataOTP", "firmaQualificata"]
    selected_signature = signature_types[int(id_area) - 1]
    
    completed_processes = get_completed_processes()
    filtered_processes = [p for p in completed_processes if p['signature'] == selected_signature]
    times = [p['duration'] for p in filtered_processes]
    
    # Parametri di ricerca e ordinamento
    search_query = request.args.get('search', '').lower()
    sort_by = request.args.get('sort', 'start_date')
    order = request.args.get('order', 'asc')
    page = int(request.args.get('page', 1))

    filtered_list, has_next_page, total_pages = filter_sort_paginate(
        filtered_processes,
        search_query=search_query,
        sort_by=sort_by,
        order=order,
        page=page
    )

    plot, _ = create_density_plot(filtered_processes, selected_signature)
    plot.x_range.start = min(times) if times else 0
    #plot.x_range.end = max(times) * 1.1 if times else 1

    script, div = components(plot)

    return render_template(
        'details.html',
        script=script,
        div=div,
        processes=filtered_list,
        signature=selected_signature,
        page=page,
        total_pages=total_pages,
        sort=sort_by,
        order=order,
        search=search_query,
        has_next_page=has_next_page,
    )

if __name__ == '__main__':
    app.run(debug=True)
