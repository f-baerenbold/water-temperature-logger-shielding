# water-temperature-logger-shielding
Analysis and results of an experiment to quantify the effect of solar radiation on lake/pond water temperature loggers.

#### Beware ####
We are not allowed to make the meteo data available on the github. Thus, there is only a sample of the meteo data on github (1 day). The data can, however, be downloaded for free following this procedure:

1. go to https://gate.meteoswiss.ch/idaweb/login.do
2. click on "Register now" and follow the instructions
3. you should receive a login in the next 1-2 days
4. if you have the login => log in
5. go to "Station portal (left side menu). If necessary set the language to English (top of the webpage)
6. type "LUZ" into the field "Short name" and click "search"
7. Toggle the station with the short name "LUZ" (very right side)
8. Click on parameter preselection and toggle the following 4 parameters: pva200s0, gre000z0, tre200s0 and fkl010z0.
This can be done by searching each parameter in the "short name" and then toggle it individually.
9. Click on time preselection and choose the whole year 2022 (or just from May to September 2022)
10. Click on data inventory, check that all parameters are there and toggle them (very right side)
11. Order the data by choosing "csv" as "Datenformat" and give the order a name
12. Check again the "summary"
13. accept the general terms and conditions and order definitively
14. After 1-2 minutes you should be able to click on the download link

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