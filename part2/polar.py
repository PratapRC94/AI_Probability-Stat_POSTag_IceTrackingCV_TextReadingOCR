from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)

def calculateTranspose(edgeMatrix):
    transposeMatrix = [[edgeMatrix[j][i] for j in range(len(edgeMatrix))] for i in range(len(edgeMatrix[0]))]
    return transposeMatrix

# This considers that the boundary is smooth
#  It decreases the probability of pixel as it goes far from the previous pixel
def calculateTransitionProbabilities( index , arrayList ):
    result = []

    result =[0]*len(arrayList)
    # assuming probabilites in range of 15, else giving 0 probability to rest indexes
    for i in range(index-15,index-10):
        if( i < len(result) and i >= 0 ):
            result[i]=0.1

    for i in range(index-10,index-5):
        if( i < len(result) and i >= 0 ):
            result[i]=0.25
    
    for i in range(index-5,index-2):
        if( i < len(result) and i >= 0 ):
            result[i]=0.35
    
    for i in range(index-2,index+2):
        if( i < len(result) and i >0 ):
            result[i]=0.5

    for i in range(index+2,index+5):
        if( i < len(result) and i >0 ):
            result[i]=0.25
    
    for i in range(index+5,index+10):
        if( i < len(result) and i >0 ):
            result[i]=0.2
    
    for i in range(index+10,index+16):
        if( i < len(result) and i >0 ):
            result[i]=0.1

    return result

# emission probabilites, normalising the edge matrix
def calculateEmissionProbabilities( arrayList):
    maximum = max(arrayList)
    minimum = min(arrayList)
    # normalising the edge probabilities to be in range 0-1
    for i in range( len(arrayList)):
        arrayList[i]=(arrayList[i]-minimum)/(maximum-minimum)

    return arrayList

# getting boundary by applying viterbi algorithm
def viterbi( matrix ):
    # initial point maximum among the column
    result = [matrix[0].index(max(matrix[0]))]
    # initial probability as 1
    p = 1
    # calculating boundary till m columns
    for i in range(1,len(matrix)):
        transitionProbabilities = calculateTransitionProbabilities(result[i-1],matrix[i])
        emissionProbabilities = calculateEmissionProbabilities(matrix[i])
        stateOutput=[]
        # checking for maximum probability pixel by viterbi algorithm
        for j in range(len(transitionProbabilities)):
            stateOutput.append(transitionProbabilities[j]*emissionProbabilities[j]*p)
        
        p = max(stateOutput)
        # the index of pixel with maximum probability goes to result
        result.append(stateOutput.index(p))
    
    return result

# getting boundary taking initial point as human provided
def humanCoordinates( row, column , matrix ):
    result = [0]*len(matrix)
    # first point given by human
    result[column] = row
    # initial probability set as 1
    p = 1

    # taking for columns forward to human-providen coordinates
    for i in range( column+1 , len(matrix) ):
        transitionProbabilities = calculateTransitionProbabilities(result[i-1],matrix[i])
        emissionProbabilities = calculateEmissionProbabilities(matrix[i])
        stateOutput= [0]*len(transitionProbabilities)
        for j in range(len(transitionProbabilities)):
            stateOutput[j] = ( transitionProbabilities[j] * emissionProbabilities[j] * p )
        
        p = max(stateOutput)
        result[i] = stateOutput.index(p)

    # calculating for columns back from human-providen coordinates
    for i in range( column-1 , -1, -1  ):
        transitionProbabilities = calculateTransitionProbabilities(result[i+1],matrix[i])
        emissionProbabilities = calculateEmissionProbabilities(matrix[i])
        stateOutput= [0]*len(transitionProbabilities)
        for j in range(len(transitionProbabilities)):
            stateOutput[j] = ( transitionProbabilities[j] * emissionProbabilities[j] * p )
        
        p = max(stateOutput)
        result[i] = stateOutput.index(p)

    return result


# main program
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # getting transpose of matrix
    transposeMatrix = calculateTranspose(edge_strength)
    
    # FIRST PART: 
    # getting maximum of columns for airIce
    airIceSimple = []
    for i in transposeMatrix:
        airIceSimple.append(i.index(max(i)))
    # considering iceRock boundary 10 pixels down for airIce boundary, and getting maximum out
    iceRockSimple = []
    for i in range( len(airIceSimple)):
        iceRockSimple.append(transposeMatrix[i].index( max( transposeMatrix[i][airIceSimple[i]+10 : ])))


    # SECOND PART:
    # getting boundary by viterbi algorithm, for airIce
    airIceViterbi = viterbi(transposeMatrix)  
    # considering iceRock boundary 10 pixels down for airIce boundary
    subIceRockViterbiMatrix = []
    for i in range( len(airIceViterbi)):
        subIceRockViterbiMatrix.append( transposeMatrix[i][airIceViterbi[i]+10 : ])
    # calculating iceRock boundary with the new edge grid by vietrbi algorithm
    iceRockViterbi = viterbi(subIceRockViterbiMatrix)
    # getting original indexes
    for i in range(len(iceRockViterbi)):
        iceRockViterbi[i] = airIceViterbi[i] + 10 + iceRockViterbi[i]


    #  THIRD PART:
    # considering human-providen coordinates initial point with probability 1
    airIceHumanFeedback = humanCoordinates( gt_airice[1], gt_airice[0], transposeMatrix )
    iceRockHumanFeedback = humanCoordinates( gt_icerock[1], gt_icerock[0], transposeMatrix )


    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    airice_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    airice_hmm = [ image_array.shape[0]*0.5 ] * image_array.shape[1]
    airice_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]

    icerock_simple = [ image_array.shape[0]*0.25 ] * image_array.shape[1]
    icerock_hmm = [ image_array.shape[0]*0.6 ] * image_array.shape[1]
    icerock_feedback= [ image_array.shape[0]*0.75 ] * image_array.shape[1]

    write_output_image("air_ice_output.png", input_image, airIceSimple, airIceViterbi, airIceHumanFeedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, iceRockSimple , iceRockViterbi, iceRockHumanFeedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n") 