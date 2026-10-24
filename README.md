# Quant Alpha Lab

Public research site — <https://quantalphalabs.github.io>

Quantitative NFL research. Every result published, including the ones that
found nothing.

This repository contains only the generated page. It is built from a private
research repository by a script that reads the results directly, so nothing on
the site is typed in by hand and it cannot drift from what the code actually
found.

The exception is `sim/`, the snap-by-snap game simulator. Its code is in
`research/rotobot-sim/`, and `.github/workflows/snap-sim.yml` runs it on the NFL
calendar and commits the results here. The site build does not touch either folder.

Not betting advice.
