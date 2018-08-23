from fastdtw import fastdtw
import pandas as pd
from haversine import haversine
from ast import literal_eval
from gmplot import gmplot
from scipy.spatial.distance import euclidean

def sort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0][0]
        for x in array:
            if x[0] < pivot:
                less.append(x)
            if x[0] == pivot:
                equal.append(x)
            if x[0] > pivot:
                greater.append(x)
        
        return sort(less)+equal+sort(greater)  
    
    else:  
        return array

train_set = pd.read_csv('datasets/train_set.csv',converters={"Trajectory": literal_eval},index_col='tripId',nrows=20)
test_set = pd.read_csv('datasets/test_set_a1.csv',converters={"Trajectory": literal_eval})

#pairnw ena ena ta pente stoixeia apo to test_set
for traj in test_set['Trajectory']:
	#print traj
	trajs = []
	for tr in traj:
		trajs.append((tr[1],tr[2]))
	#sygkrinw me ola ta stoixeia apo to train_set
	dist = []
	for index, row in train_set.iterrows():

		points = []
		for point in row['Trajectory']:
			points.append((point[1],point[2])) #vazw sti lista tuples pou periexoun ta shmeia tis diadromis (lon,lat)
			
		distance , path = fastdtw(trajs, points, dist=haversine)
		dist.append((distance,row['journeyPatternId'],points)) #vazoume sti lista ena tuple pou periexei tin apostasi apo ton dtw to journeyPatternId kai to Trajectory

	#for  d  in dist:
	#	print d[0]
	final_dist = sort(dist)
	#for  d  in dist1:
	#	print (d[0],d[1],d[2])
	
	count = 0
	for d in final_dist:
		if count == 5 :
			break

		gmap = gmplot.GoogleMapPlotter(d[2][0][1],d[2][0][0],13)
		#print d[2][0][1]
		#print d[2][0][0]

		longitude , latitude = zip(*d[2]) 			
		#print latitude
		#print longitude

		gmap.plot(latitude, longitude, 'cornflowerblue', edge_width=10)

		# Write the map in an HTML file
		str_merge = 'Neighbor' + str(count) + '.html'
		gmap.draw(str_merge)

		count+=1		

	break