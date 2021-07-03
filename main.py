"""
"Main" and does not contain class definitions. Sandbox, playground, testing site...
Whatever name that floats your boat!
"""

from control_panel import ControlPanel

pccm_instance = ControlPanel()
pccm_instance.init_ds()
pccm_instance.init_mf()
pccm_instance.fit()
pccm_instance.print_df()

pccm_instance.plot()
