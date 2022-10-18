#This script creates the sidebar that credits us with the dashboard

#Import Libraries
import streamlit as st

#Sidebar Function Definition
def Sidebar_Main():
    #Shows BYU Logo and presents dashboard
    st.sidebar.image('images/BYU_Logo_White.png')
    st.sidebar.title("This Mushroom Dashboard is brought to you by the 3 Amigos.")

    #Shows Joey's image and name
    st.sidebar.image('images/Joey.png')
    st.sidebar.write('Joey DelaCrazy')

    #Shows Meysa's image and name
    st.sidebar.image('images/Meysa.png')
    st.sidebar.write('Meysa Mulford')

    #Shows Brigham's image and name
    st.sidebar.image('images/Brigham.png')
    st.sidebar.write('Brigham Perry')
