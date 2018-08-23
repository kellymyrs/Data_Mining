from gmplot import gmplot
import pandas as pd
from ast import literal_eval


train_set = pd.read_csv('datasets/train_set.csv',converters={"Trajectory": literal_eval},index_col='tripId')

#divazoume ta journey pattern id gia na vroume 5 diaforetika
count = 0 #plithos twn diadromwn pou tha vgaloume html arxeio
cache_jpid = [] #lista me ta journeyPatternId twn diadromwn pou tha anaparastisoume

#gia na paroume parapanw apo mia stiles xrhsimopoioyume auti ti for
for index, row in train_set.iterrows():
	if count == 5:
		break

	#pairnoume tous 4 prwtous xaraktires wste na vroume 5 diaforetikes diadromes
	first_four_str = row['journeyPatternId'][:4]
	if first_four_str not in cache_jpid:
		if row['Trajectory'] != []:		
			cache_jpid.append(first_four_str)
			final_points = [] #lista me ta shmeia tis diadromis

						
			#print row['Trajectory']
			for lists in row['Trajectory']:
				final_points.append((lists[1],lists[2])) #vazw sti lista tuples pou periexoun ta shmeia tis diadromis (lon,lat)
			#print final_points
			

			gmap = gmplot.GoogleMapPlotter(final_points[0][1],final_points[0][0],13)
			#print final_points[0][0]
			#print final_points[0][1]

			longitude , latitude = zip(*final_points) 			
			#print latitude
			#print longitude

			gmap.plot(latitude, longitude, 'cornflowerblue', edge_width=10)

			# Write the map in an HTML file
			str_merge = 'map' + str(count) + '.html'
			gmap.draw(str_merge)

			count += 1