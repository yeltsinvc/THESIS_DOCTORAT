# -*- coding: utf-8 -*-
"""
Created on Wed May 26 00:36:52 2021

@author: valero
"""

import plotly.express as px
import plotly.io as pio

long_df = px.data.medals_long()

fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
fig.show()
pio.write_html(fig, file="index.html", auto_open=True)