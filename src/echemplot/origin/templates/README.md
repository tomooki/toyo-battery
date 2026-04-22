# Origin graph templates

`echemplot.origin.push_to_origin` creates graphs from three Origin
graph templates ported from the legacy `TOYO_Origin_2.01` pipeline:

| Template | Role |
| --- | --- |
| `charge_discharge.otpu` | V vs Q charge/discharge curves |
| `cycle_efficiency.otpu` | Per-cycle discharge capacity + CE dual-Y |
| `dqdv.otpu` | dQ/dV vs V curves |

These files ship with the wheel and are loaded automatically by
`push_to_origin`. No additional setup is required for the default path.

## Override path

If you want to substitute your own templates, set the environment
variable `ECHEMPLOT_ORIGIN_TEMPLATE_DIR` to a directory containing the same
three filenames. `push_to_origin` resolves templates in this order:

1. `$ECHEMPLOT_ORIGIN_TEMPLATE_DIR/{filename}` (when the env var is set)
2. The bundled file inside this package directory

If a required template is missing at runtime, `push_to_origin` raises
`FileNotFoundError` with a message naming both remediation paths.
