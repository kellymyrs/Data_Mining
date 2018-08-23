import pandas as pd
from haversine import haversine
from ast import literal_eval
from gmplot import gmplot

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

def e_lcss(t0, t1, eps):
    n0 = len(t0)
    n1 = len(t1)
    path = []
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
        	#print "harvesine"
        	print haversine(t0[i - 1], t1[j - 1])
            if haversine(t0[i - 1], t1[j - 1]) <= eps:
            	C[i][j] = C[i - 1][j - 1] + 1
            	#path.append((t0[i - 1][1],t0[i - 1][0]))
            	#print "---"
            	#print t0[i-1]
            	#print t1[j-1]
            	#print "---"
            	path.append(t0[i-1])
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
	lcss = 1 - float(C[n0][n1]) / min([n0, n1])
	print "-----path of lcss----------"
	print lcss
	return lcss, path


train_set = pd.read_csv('datasets/train_set.csv',converters={"Trajectory": literal_eval},index_col='tripId',nrows=20)
test_set = pd.read_csv('datasets/test_set_a2.csv',converters={"Trajectory": literal_eval})

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
		#print "--------------------------"
		#print "inputs"
		#print trajs
		#print points	
		distance , path = e_lcss(trajs, points, 200)
		dist.append((distance,path,row['journeyPatternId'],points)) #vazoume sti lista ena tuple pou periexei tin apostasi apo ton dtw to koino path pou mas epistrefei o dtw to journeyPatternId kai to Trajectory
		#print "-----------------------------"
		#print "1111111111111111111111111"
		#print path
		#print "122222222222222222222222222211111111111111111111"

	#for  d  in dist:
	#	print d[0]
	final_dist = sort(dist)
	#for  d  in dist1:
	#	print (d[0],d[1],d[2])
	
	count = 0
	generic = 0
	for d in final_dist:
		if count == 5 :
			break

		gmap = gmplot.GoogleMapPlotter(d[3][0][1],d[3][0][0],13)
		#print d[2][0][1]
		#print d[2][0][0]

		longitude , latitude = zip(*d[3])
		#print "-------------------------lat1------------------------"
		#print longitude
		#print "-------------------------end of lat 1-----------------"
		longitude2, latitude2 = zip(*d[1]) 	
		#print "---------------------------lat2-------------------------"
		#print longitude2		
		#print "---------------------------end of lat 2-------------------"
		#print latitude2
		#print latitude
		#print longitude

		gmap.plot(latitude, longitude, 'cornflowerblue', edge_width=10)


		gmap.plot(latitude2,longitude2, 'red' ,edge_width=10)

		# Write the map in an HTML file
		str_merge = 'Neighbor' + str(count+1)+ "_gen_" +str(generic+1) + '.html'
		gmap.draw(str_merge)
		generic += 1

		count+=1		

	#break