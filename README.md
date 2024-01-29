# water-temperature-logger-shielding
Analysis and results of an experiment to quantify the effect of solar radiation on lake/pond water temperature loggers.

The pond script can be executed by typing

~~~bash
python ./scripts/pond.py
~~~

At the start of the script, one can choose whether only data from 1 week in june (used for time series plots) or for the whole study period (used for scatter plots) should be used
Set "period" to "all" or "june"

The lake script can be executed in the same way (replacing pond.py by lake.py)

The heat flux script can be executed by typing

~~~bash
./scripts/heat_budget.py
~~~

Again, one can set "period" to "all" or to "june". In addition, if the switch "add_10_degrees" is set to True then 10 °C will be added to the pond temperature (see manuscript for details).

First version: 29.01.2024 by Fabian Bärenbold, Eawag