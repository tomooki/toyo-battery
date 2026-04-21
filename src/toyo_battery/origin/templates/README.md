# Origin graph templates

`toyo_battery.origin.push_to_origin` creates graphs from three Origin
graph templates shipped by the legacy `TOYO_Origin_2.01` pipeline:

| Template | Role |
| --- | --- |
| `charge_discharge.otpu` | V vs Q charge/discharge curves |
| `cycle_efficiency.otpu` | Per-cycle discharge capacity + CE dual-Y |
| `dqdv.otpu` | dQ/dV vs V curves |

## Why this directory is empty

The `.otpu` files are proprietary artifacts from the v2.01 pipeline. We
have not yet cleared their license for inclusion in this open-source
package, so this directory ships **without** them. Follow-up tracked in
the P5 issue.

## Where to put the templates

`push_to_origin` resolves the template path in this order:

1. If the environment variable `TOYO_ORIGIN_TEMPLATE_DIR` is set, it is
   used as the directory (e.g.
   `set TOYO_ORIGIN_TEMPLATE_DIR=C:\Origin\MyTemplates`).
2. Otherwise, the files are looked up inside this package directory
   (`toyo_battery/origin/templates/`).

If a required template is missing at runtime, `push_to_origin` raises
`FileNotFoundError` with a message naming both remediation paths.
