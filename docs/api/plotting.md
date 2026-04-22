# Plotting

Two interchangeable backends expose the same three functions —
`plot_chdis`, `plot_cycle`, `plot_dqdv` — differing only in the figure
type they return. Pick the backend based on whether you want static
Matplotlib figures (PNG / PDF via `savefig`) or interactive Plotly
figures (HTML / PNG via `write_image`).

## Matplotlib backend

Requires the `[plot]` extra.

::: toyo_battery.plotting.matplotlib_backend

## Plotly backend

Requires the `[plotly]` extra (`plotly` + `kaleido`).

::: toyo_battery.plotting.plotly_backend
