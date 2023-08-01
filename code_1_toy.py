import os
import csv
import sys
import random
import time
import json
import shapely.wkt
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
from shapely.geometry import Point
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians

# The number of pois = 17
# Calculation process: Galveston County 5330 pois
# 980 km^2 land area --> toy network area 3.14km^2
# num_pois = round(5330/980*3.141592) #17
# num_agents = 10

def make_circle_agent(num_pois,num_agents,rseed):
    
    circle_address = "/mnt/d/TRB-CT/ToyNetwork/point_1km_10000.shp"

    poi_circle_json_address = "/home/sangung/PostDisasterSim/model/code_1_POI_node/code_1_output_poi.json"
    poi_circle_address = "result/code_1_output_poi_circle_{}_{}.json".format(rseed,num_pois)
    poi_circle_adj_json_address = "result/code_2_output_poi_circle_adj_{}_{}.json".format(rseed,num_pois)
    poi_circle_activity_address = "/home/sangung/PostDisasterSim/model/code_3_POI_dynamic/code_3_output_poi_activity.json"
    poi_circle_activity_address_2 = "result/code_3_output_poi_acitvity_circle_{}_{}.json".format(rseed,num_pois)

    home_circle_json_address = "/home/sangung/PostDisasterSim/model/code_4_Mobility_node/code_4_output_house.json"
    home_circle_address = "result/code_4_output_house_circle_{}_{}.json".format(rseed,num_agents)
    home_circle_adj_address = "result/code_5_output_house_circle_edge_{}_{}.json".format(rseed,num_agents)
    home_circle_activity_address = "/home/sangung/PostDisasterSim/model/code_6_Mobility_dynamic/code_6_human_activity.json"
    home_circle_activity_address_2 = "result/code_6_human_activity_circle_{}_{}.json".format(rseed,num_agents)
    
                                           

    #code_1
    point_circle_gpd = gpd.read_file(circle_address)

    poi_circle = point_circle_gpd.sample(n=num_pois)
    point_circle_gpd = point_circle_gpd.loc[[i not in poi_circle.index for i in point_circle_gpd.index],:]

    sss = {'id':[str(i) for i in poi_circle.index.to_list()],
           'lon':poi_circle["geometry"].x.to_list(),
           'lat':poi_circle["geometry"].y.to_list(),
           'county': ['Toy' for i in range(len(poi_circle))]
          }
    
    with open(poi_circle_address, "w") as outfile:
        json.dump(sss, outfile)

    #code_2
    with open(poi_circle_address, "r") as f:
        df_poi_data = json.load(f)
                                           

    time1 = time.time()
    id_list, lon_list, lat_list = list(df_poi_data["id"]), list(df_poi_data["lon"]), list(df_poi_data["lat"])
    #build the mapping from fixed locations to user id
    #{(xxx 0.02,yyy 0.02):[0,1,2,100]}
    mapping = dict()
    print ("# row", len(id_list))
    for i in range(len(id_list)):
        if i % 10000 == 0:
            print (i)
        lon, lat = abs(lon_list[i]), abs(lat_list[i])
        lon_round, lat_round = int(lon*100.0)/100, int(lat*100.0)/100
        key = (lon_round, lat_round)
        key_1 = (lon_round + 0.01, lat_round)
        key_2 = (lon_round + 0.01, lat_round + 0.01)
        key_3 = (lon_round, lat_round + 0.01)
        key_4 = (lon_round - 0.01, lat_round + 0.01)
        key_5 = (lon_round - 0.01, lat_round)
        key_6 = (lon_round - 0.01, lat_round - 0.01)
        key_7 = (lon_round, lat_round - 0.01)
        key_8 = (lon_round + 0.01, lat_round - 0.01)
        key_list = [key, key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8]
        for one_key in key_list:
            if one_key not in mapping:
                mapping[one_key] = [i]
            else:
                mapping[one_key].append(i)
    print (len(mapping))
    print (np.sum([len(mapping[key]) for key in mapping]))
    time2 = time.time()
    print ("the total running time", time2-time1)

    #2.2 compute the spatial distance (unit: km) between locations 1 and 2.
    #input: loc1, loc2
    #output: distance
    def distancePair(loc1,loc2):    #loc1:[lon1,lat1]; loc2:[lon2,lat2]
        R = 6373.0
        lon1 = radians(loc1[0])
        lat1 = radians(loc1[1])
        lon2 = radians(loc2[0])
        lat2 = radians(loc2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        return distance

    #2.3 compute the spatial distance (unit: km) between locations 1 and 2.
    #input: mapping
    #output: pair_within_one_mile
    time1 = time.time()
    #second search every set
    pair_within_one_mile = list()
    edge_idx, key_idx = 0, 0
    print ("# index: ", len(mapping))
    for key in mapping:
        key_idx += 1
        idx_list = mapping[key]  
        for idx1 in idx_list:
            for idx2 in idx_list:
                if idx1 < idx2:
                    loc1, loc2 = [lon_list[idx1], lat_list[idx1]], [lon_list[idx2], lat_list[idx2]]
                    if distancePair(loc1, loc2) < 1.000:            # 1 km
                        pair = str(idx1) + "_" + str(idx2)
                        pair_within_one_mile.append(pair)
                        edge_idx += 1
        if key_idx % 100 == 0:
            print ("key_idx: ", key_idx)
            time2 = time.time()
            print ("the total running time until now", time2 - time1)
            print ("---------------------------------------------")


    #2.4 statistics of pair_within_one_mile.        
    print ("# pair before considering repeating", len(pair_within_one_mile))
    pair_within_one_mile_list = list(set(pair_within_one_mile))
    print ("# pair after deleting repeating", len(pair_within_one_mile_list))
    output_dict = dict()
    for i in range(len(pair_within_one_mile_list)):
        output_dict[i] = pair_within_one_mile_list[i]

    with open(poi_circle_adj_json_address, "w") as outfile:
        json.dump(output_dict, outfile)

    with open(poi_circle_json_address, "r") as f:
        df_poi = pd.DataFrame.from_dict(json.load(f),orient='columns')

    with open(poi_circle_activity_address, "r") as f:
        ac_poi = json.load(f)
    
    #Sample_pois
    # id_poi_sample = df_poi.loc[df_poi.county == 'Galveston County',].sample(num_pois).id.values.tolist() # Random
    id_poi_sample = df_poi.loc[df_poi.county == 'Galveston County',].id.values.tolist()[0:num_pois] # Stacking
    ac_poi_sample = dict()
    id_list = list(df_poi_data["id"])
    for i in range(len(id_poi_sample)):
        id_poi = id_list[i]
        ac_poi_sample[int(id_poi)]=ac_poi[id_poi_sample[i]]
        
    with open(poi_circle_activity_address_2,"w") as outfile:
        json.dump(ac_poi_sample,outfile)
        
    
    

    #code_4


    #input: path
    agents_circle_gpd = point_circle_gpd.sample(n=num_agents)
    point_circle_gpd = point_circle_gpd.loc[[i not in agents_circle_gpd.index for i in point_circle_gpd.index],:]

    sss = {'id':[str(i) for i in agents_circle_gpd.index.to_list()],
           'lon':agents_circle_gpd["geometry"].x.to_list(),
           'lat':agents_circle_gpd["geometry"].y.to_list(),
           'county': ['Toy' for i in range(len(agents_circle_gpd))]
          }
    with open(home_circle_address, "w") as outfile:
        json.dump(sss, outfile)


    #code_5


    with open(home_circle_address, "r") as f:
        read_file = json.load(f)
    id_list = list(read_file["id"])
    lon_list = list(read_file["lon"])
    lat_list = list(read_file["lat"])

    #2.1 #map the lon_list and lat_list to {(xxx 0.02,yyy 0.02):[0,1,2,100]}
    #input: id_list, lon_list, lat_list
    #output: mapping
    time1 = time.time()
    mapping = dict()
    for i in range(len(id_list)):
        #if i %10000 == 0:
        #    print (i)
        lon, lat = abs(lon_list[i]), abs(lat_list[i])
        lon_round, lat_round = int(lon*50.0)/50, int(lat*50.0)/50
        key = (lon_round, lat_round)
        key_1 = (lon_round + 0.02, lat_round)
        key_2 = (lon_round + 0.02, lat_round + 0.02)
        key_3 = (lon_round, lat_round + 0.02)
        key_4 = (lon_round - 0.02, lat_round + 0.02)
        key_5 = (lon_round - 0.02, lat_round)
        key_6 = (lon_round - 0.02, lat_round - 0.02)
        key_7 = (lon_round, lat_round - 0.02)
        key_8 = (lon_round + 0.02, lat_round - 0.02)
        key_list = [key, key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8]

        for one_key in key_list:
            if one_key not in mapping:
                mapping[one_key] = [i]
            else:
                mapping[one_key].append(i)
    print (len(mapping))
    print (np.sum([len(mapping[key]) for key in mapping]))
    time2 = time.time()
    print ("the total running time", time2-time1)


    def distancePair(loc1,loc2):    #loc1:[lon1,lat1]; loc2:[lon2,lat2]
        R = 6373.0
        lon1 = radians(loc1[0])
        lat1 = radians(loc1[1])
        lon2 = radians(loc2[0])
        lat2 = radians(loc2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        return distance

    #2.3. 
    #input: mapping
    #output: pair_within_one_mile
    time1 = time.time()  #second search every set
    pair_within_one_mile = list()
    edge_idx = 0
    key_idx = 0
    for key in mapping:
        key_idx += 1
        idx_list = mapping[key]  
        for idx1 in idx_list:
            for idx2 in idx_list:
                if idx1 < idx2:
                    loc1 = [lon_list[idx1], lat_list[idx1]]
                    loc2 = [lon_list[idx2], lat_list[idx2]]
                    if distancePair(loc1, loc2) < 1.609:
                        pair = str(idx1)+"_"+str(idx2)
                        pair_within_one_mile.append(pair)
                        edge_idx += 1
        if key_idx %10 == 0:
            print ("key_idx:", key_idx, "total_idx:", len(mapping))
            print ("size of set:", len(idx_list))
            print ("total edge:", edge_idx)
            time2 = time.time()
            print ("the total running time until now", time2 - time1)
            print ("---------------------------------------------")

    #2.4
    #input: pair_within_one_mile
    #output: output_dict, pair_within_one_mile_list
    print (len(pair_within_one_mile))
    pair_within_one_mile_list = list(set(pair_within_one_mile))
    print (len(pair_within_one_mile_list))

    output_dict = dict()
    for i in range(len(pair_within_one_mile_list)):
        output_dict[i] = pair_within_one_mile_list[i]
    print (len(pair_within_one_mile_list))

    with open(home_circle_adj_address, "w") as outfile:
        json.dump(output_dict, outfile)


    #code_6

    #1.1. read user homes
    with open(home_circle_address, "r") as f:
        human_location = json.load(f)
    # print(len(human_location))
    # print(len(human_location["id"]))
    # print(human_location["id"][0])
    # print(human_location["lon"][0])
    # print(human_location["lat"][0])
    # print(human_location["county"][0])

    #{"123": [-100, 25]}
    human_lon_lat_dict = {human_location["id"][i]:[human_location["lon"][i], human_location["lat"][i]] for i in range(len(human_location["lat"]))}
    #2.2. initilize the activity_level
    #input: human_location
    #output: activity level
    activity_level = dict()
    for user in human_location["id"]:
        activity_level[user] = [[0 for i in range(31)], [0 for i in range(30)]]
    # print (activity_level[human_location["id"][0]])
    # print (len(activity_level))

    

    #Sample_human
    with open(home_circle_json_address, "r") as f:
        df_human = pd.DataFrame.from_dict(json.load(f),orient='columns')

    with open(home_circle_activity_address, "r") as f:
        ac_human = json.load(f)

    with open(home_circle_address, "r") as f:
        df_human_data = json.load(f)
    
    
    id_human_sample = df_human.loc[df_human.county == 'Galveston County',].sample(num_agents,replace=True).id.values.tolist()
    # id_human_sample = df_human.loc[df_human.county == 'Galveston County',].id.values.tolist()[0:num_agents] # Stacking
    ac_human_sample = dict()
    id_list = list(df_human_data["id"])
    for i in range(len(id_human_sample)):
        id_human = id_list[i]
        ac_human_sample[int(id_human)]=ac_human[id_human_sample[i]]
    
#     for i in id_human_sample:
#         ac_human_sample[i] = ac_human[i]
    
    with open(home_circle_activity_address_2,"w") as outfile:
        json.dump(ac_human_sample,outfile)
    print('End writing code_6_human_activity_circle.json')
        