import time,os,json

class  map:

	map_key = []
 
	def __init__(self , n_row , n_column , sub_matrix_length ) -> None:
		
		self.map_key = self.get_map_formatted(n_row,n_column,sub_matrix_length)
	
		pass

	def paint_cell(self, row,column):
		
		return self.stockastic_function(row,column)

	def paint_row_true(self, row_length):
		
		row102 = []
		for i in range(0,row_length):
			row102.append(True)
			
		return row102

	def paint_function( self, row_length=100 , column_length=100):
		
		matrix102x102 = []
		
		for i in range(0,row_length):
			
			row102 = []
			if i == 0:
				row102 = self.paint_row_true(row_length)
				
			elif i == row_length - 1:
				row102 = self.paint_row_true(row_length)
				
			else:   
			
				for j in range(0,column_length):
					
					if j == 0:
						row102.append(True)
						continue
					elif j == column_length - 1:
						row102.append(True)
						continue
					
					cell = self.paint_cell(i,j)
			 	
					row102.append(cell)
					
					pass    
			
			matrix102x102.append(row102)
		
		return matrix102x102

	def get_map( self,row,column,row_sub_matrix_length):
		
		game_map = []
		row += 1
		column += 1
		
		for i in range(1,row):
			
			my_row = []
			for j in range(1,column):
				
				matriz102x102 = self.paint_function( row_sub_matrix_length , row_sub_matrix_length)

				my_row.append( { "row": i , "column": j , "matrix": matriz102x102 } )
				pass
			
			game_map.append(my_row)
		
		return game_map

	def formatting_matrix(self, map  , row_length , column_length , row_sub_matrix_length ):
		
		result = []
		for i in range( 0, row_length):
			
			for sub_i in range( 0 , row_sub_matrix_length ):
				
				row = []
				for j in range(0 , column_length):
					
					sub_matrix = map[i][j]["matrix"]
					row += sub_matrix[sub_i]
					pass
				
				result.append(row)
				
				pass
		
		return result

	def get_map_formatted( self, row,column , sub_matrix_length):
		
		map = self.get_map(row,column,sub_matrix_length)
		format = self.formatting_matrix(map,row,column,sub_matrix_length)
		
		return format
	
	def stockastic_function(self,row,column):
	
		# Get the current time in milliseconds
		current_time_msec = int(time.time() * 1000)

		seed_ = 300
		offset = 0
		
		if row + column > seed_ or row == 1 or column == 1:
			offset = 1
			pass
		else:
			offset = -1
			pass

		if (current_time_msec % row == 0 and not current_time_msec % column == 0) or \
		(current_time_msec % (column + offset) == 0 and \
			not current_time_msec % (row + offset) == 0):
			
			return False

		return True

def comprise_data(map:list):
    
    comprised_map = []
    
    len_row = 0
    len_column = 0
    
    for i,row in enumerate(map):
        
        my_row = ""
        for j,cell in enumerate(row):
            if not cell:
                my_row += f" {j}"
        
        comprised_map.append(my_row)
    
    len_row = len(map)
    len_column = len(map[0])
	
    return comprised_map , len_row , len_column

def generate_dataset():
    
	k=0
	while k < 3:
		os.system("cls")
		print(f"\033[1;32m number of maps generated: \033[1;31m {k} \033[0m")
	
		x = map( 18 ,18 , 57 )

		comp_map , len_row , len_column = comprise_data(x.map_key)
	
		data = { "row": len_row , "column": len_column }

		i=0
		
		for row in comp_map :
			data[i] = row
			i+=1
		
		file_path = f'../MapsSet/map{k}.json'

		# Open the file in write mode ('w')
		with open( file_path , 'w') as file:
			# Serialize the data to JSON and write it to the file
			json.dump(data, file, indent=4)
		
		file.close()
	
		k += 1

def extract_row_column( json_data ):
    
    len_row = json_data["row"]
     
    len_column = json_data["column"]
    
    return len_row , len_column

