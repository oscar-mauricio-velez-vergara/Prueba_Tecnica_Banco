#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sparky_bc import Sparky
import pandas as pd
import platform

sp = Sparky(username='omvelez',  dsn='impala-pro')

archivo = r'C:\Users\omvelez\OneDrive - Grupo Bancolombia\Solicitudes_Gerencia\PRUEBA TECNICA\REPORTES\alertas 1.csv'

df = pd.read_csv(archivo)  
sp.subir_df(df, 'proceso_pana_cumpl.omvelez_archivo_prueba', modo="overwrite")


# In[ ]:


archivo2 = r'C:\Users\omvelez\OneDrive - Grupo Bancolombia\Solicitudes_Gerencia\PRUEBA TECNICA\REPORTES\Resultado_Prueba_Tecnica.xlsx'

df1 = pd.read_excel(archivo2)  
sp.subir_df(df1, 'proceso_pana_cumpl.omvelez_resultado_modelo', modo="overwrite")

