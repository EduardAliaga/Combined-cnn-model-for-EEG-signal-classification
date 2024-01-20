"""Make plots of results."""
import plotly.graph_objects as go

data = [
    ("Disc LogR", 3907, 0.817, "square"),
    ("Disc EEGNet", 12274, 0.930, "circle"),
    ("Disc 1D CNN", 542210, 1.103, "cross"),
    ("Disc 2D CNN", 1603042, 1.153, "x"),
    ("Disc combined cnn tiny", 27890, 1.137, "pentagon"),
    ("Disc combined cnn", 168370, 1.299, "pentagon"),
    ("Gen LDA", 819401, 0.678, "diamond"),
    ("Gen LogR", 819401, 0.218, "star-triangle-up"),
]

highlighted_model = "Disc combined cnn"  # Model to highlight

fig = go.Figure()
fig.update_layout(
    template="simple_white",
    xaxis_title="Parameter Count",
    yaxis_title="ITR (bits per symbol)",
    legend=dict(
        orientation="v",
        yanchor="auto",
        y=0,
        xanchor="auto",
        x=1,
        borderwidth=2,
        font=dict(size=10),  # Smaller font size for the legend
    ),
    font=dict(size=20),
    autosize=False,
    height=400,
    width=600,
    margin=dict(l=50, r=10, b=50, t=10, pad=4),
)

# Add the traces for each model
for model, params, itr, symbol in data:
    marker_size = 20 if model == highlighted_model else 15
    marker_line_width = 3 if model == highlighted_model else None
    marker_line_color = 'black' if model == highlighted_model else None
    
    fig.add_trace(
        go.Scatter(
            x=[params],
            y=[itr],
            mode="markers",
            marker=dict(
                size=marker_size,
                symbol=symbol,
                line=dict(width=marker_line_width, color=marker_line_color),
            ),
            name=model
        )
    )

fig.write_image("itr-vs-params.png")
fig.show()
