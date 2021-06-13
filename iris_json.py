#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:11:43 2021

@author: halidziya
"""
import pandas as pd
import json

filename = '/home/halidziya/Downloads/iris.data'

data = pd.read_csv(filename)

parsed = [row.tolist() for i, row in data.iterrows()]

with open('iris.json', 'w') as fil:
    json.dump(parsed, fil)
    
    
"""

function draw(data)
{
    for(i=0;i<data.length;i++)
    {
        if (data[i][5] == 'Iris-setosa'):
            color = "green"
        if (data[i][5] == 'Iris-versicolor'):
            color = "blue"
        if (data[i][5] == 'Iris-virginica'):
            color = "red"
        addCircle(data[i][0], data[i][1], color);
    }

}

$( document ).ready( function (){
$.get('iris.json', draw)
});
"""