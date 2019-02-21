import aligned_helper
import sys
import copy
import math

BLACK_INDEX = 0
WHITE_INDEX = 255
MAX_DIST = 2
# filter for detect_edges function
AVG_FILTER_DE = [[-1/8,-1/8,-1/8],[-1/8,1,-1/8],[-1/8,-1/8,-1/8]]
# filter that take the average of the pixels in square 3*3
AVG_FILTER = [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
NUM_OF_PARAM = 3
ERROR_MESSAGE = "Wrong number of parameters. The correct usage is: \n " \
                "aligned.py <image_source> <output > <max_diagonal>"


def otsu(image):
    """Function that gets an image and return optimal threshold value"""
    var = 0
    best_threshold = 0
    for threshold in range(256):
        num_of_blacks, num_of_whites, sum_of_blacks, sum_of_whites = (0, 0,
                                                                      0, 0)
        for row in image:
            for column in row:
                if column < threshold:
                    num_of_blacks += 1
                    sum_of_blacks += column
                else:
                    num_of_whites += 1
                    sum_of_whites += column
        if num_of_blacks == 0 or num_of_whites == 0:
            # We dont like to divide by zero
            continue
        mean_black = sum_of_blacks / num_of_blacks
        mean_whites = sum_of_whites / num_of_whites
        new_var = num_of_blacks * num_of_whites * ((mean_black -
                                                   mean_whites)**2)
        # This formula of variance is taken from the lecture
        if new_var > var:
            var = new_var
            best_threshold = threshold
    return best_threshold


def threshold_filter(image):
    """This function gets an image and return a new image in the same size,
    but the pixels change to be black or white, depend the variance"""
    new_image = copy.deepcopy(image)
    var = otsu(new_image)
    for threshold in range(256):
        for index_row, row in enumerate(new_image):
            for index_col, column in enumerate(row):
                if column < var:
                    new_image[index_row][index_col] = BLACK_INDEX
                else:
                    new_image[index_row][index_col] = WHITE_INDEX
    return new_image


def corners(filter, new_image):
    """Function that gets 2 parameters - filter, and image, and return a fix
    value pixel of all the 4 pixels in the corners"""
    height = len(new_image)
    width = len(new_image[0])
    left_up_value, right_up_value, left_down_value, right_down_value = (0, 0,
                                                                 0, 0)
    for ro in range(-1,2): # left-up corner
        for co in range(-1,2):
            if ro < 0 or co < 0:
                left_up_value = new_image[0][0]*filter[ro][co] + left_up_value
            else:
                left_up_value = new_image[ro][co]*filter[ro][co] + \
                                left_up_value

    for ro in range(-1,2): # right-up corner
        for co in range(-1,2):
            if ro < 0 or width - 1 + co >= width:
                right_up_value = new_image[0][width-1]*filter[ro][co] + \
                                 right_up_value
            else:
                right_up_value = new_image[ro][width - 1 +co]*filter[ro][co]\
                                 + right_up_value

    for ro in range(-1,2): # left-down corner
        for co in range(-1,2):
            if height - 1 + ro >= height or co < 0:
                left_down_value = new_image[height-1][0]*filter[ro][co]\
                                 + left_down_value
            else:
                left_down_value = new_image[height - 1 + ro][co] * filter[ro][
                    co] + left_down_value

    for ro in range(-1,2): # right-down corner
        for co in range(-1,2):
            if height - 1 + ro >= height or width - 1 + co >= width:
                right_down_value = new_image[height-1][width-1]*filter[ro][co]\
                                 + right_down_value
            else:
                right_down_value = new_image[height - 1 + ro][width - 1 + co]\
                                   * filter[ro][co] + right_down_value

    return fix_value(left_up_value), fix_value(right_up_value), fix_value(
        left_down_value), fix_value(right_down_value)


def up(col_index, filter, new_image):
    """Function that gets 3 parameters - index of column, filter, and image,
    and return a fix value pixel of all the pixels in the first row in the
    image (without the corners)"""
    new_value = 0
    for ro in range(-1,2):
        for co in range(-1,2):
            if ro < 0 :
                new_value = new_image[0][col_index]*filter[ro][co] + new_value
            else:
                new_value = new_image[ro][col_index+co] * filter[ro][co] + \
                            new_value

    return fix_value(new_value)


def down(col_index, filter, new_image):
    """Function that gets 3 parameters - index of column, filter, and image,
    and return a fix value pixel of all the pixels in the last row in the
    image (without the corners)"""
    height = len(new_image)
    new_value = 0
    for ro in range(-1,2):
        for co in range(-1,2):
            if height - 1 + ro >= height:
                new_value = new_image[height-1][col_index]*filter[ro][co] + \
                            new_value
            else:
                new_value = new_image[height - 1 + ro][col_index+co] * filter[
                    ro][co] + new_value

    return fix_value(new_value)


def left(row_index, filter, new_image):
    """Function that gets 3 parameters - index of row, filter, and image,
    and return a fix value pixel of all the pixels in the first column in the
    image (without the corners)"""
    new_value = 0
    for ro in range(-1,2):
        for co in range(-1,2):
            if co < 0:
                new_value = new_image[row_index][0] * filter[ro][
                    co] + new_value
            else:
                new_value = new_image[row_index + ro][co] * filter[ro][co] + \
                            new_value

    return fix_value(new_value)


def right(row_index, filter, new_image):
    """Function that gets 3 parameters - index of row, filter, and image,
    and return a fix value pixel of all the pixels in the last column in the
    image (without the corners)"""
    width = len(new_image[0])
    new_value = 0
    for ro in range(-1,2):
        for co in range(-1,2):
            if width - 1 + co >= width:
                new_value = new_image[row_index][width-1] * filter[ro][
                    co] + new_value
            else:
                new_value = new_image[row_index + ro][width - 1 + co] * \
                            filter[ro][co] + new_value

    return fix_value(new_value)


def middle(row_index, col_index, filter, new_image):
    """Function that gets 4 parameters - index of row, index of column, filter,
     and image, and return a fix value pixel of all the pixels in the middle of
     the image, its mean without - first and last rows, first and last
     columns, and corners"""
    new_value = 0
    for ro in range(-1,2):
        for co in range(-1,2):
            new_value = new_image[row_index + ro][col_index + co] * \
                        filter[ro][co] + new_value

    return fix_value(new_value)


def fix_value(new_value):
    """Function that gets a value of pixel and return a fix value that make
    sure that the value is in the range 0 to 255"""
    new_value = int(new_value)
    if new_value > WHITE_INDEX or new_value < -WHITE_INDEX:
        new_value = WHITE_INDEX
    if -256 < new_value < BLACK_INDEX:
        new_value = -1 * new_value
    return new_value


def apply_filter(image, filter):
    """Function that gets 2 parameters - filter and image, and return a new
    image with new pixels, depend the filter. This functions uses 5 helper
    functions - corners, up, down, left, right"""
    height = len(image)
    width = len(image[0])
    new_image = copy.deepcopy(image)

    for col_index in range(1, width-1):
        # Running on the first and last rows
        new_image[0][col_index] = up(col_index, filter, image)
        new_image[height-1][col_index] = down(col_index, filter, image)

    for row_index in range(1, height-1):
        # Running on the first and last columns
        new_image[row_index][0] = left(row_index, filter, image)
        new_image[row_index][height-1] = right(row_index, filter, image)

    for row_index in range(1, width-1):
        # Running on the middle of the image
        for col_index in range(1, height-1):
            new_image[row_index][col_index] = middle(row_index, col_index,
                                                     filter, image)
    # Manual placement of the corners:
    new_image[0][0], new_image[0][width-1], new_image[height-1][0], \
    new_image[height-1][width-1] = corners(filter, new_image)
    return new_image


def detect_edges(image):
    """Function that gets an image and return a new image that returns from
    apply_filter, with the filter: [[-1/8,-1/8,-1/8],[-1/8,1,-1/8],
    [-1/8,-1/8,-1/8]]"""
    return apply_filter(image, AVG_FILTER_DE)


def downsample_by_3(image):
    """Function that gets an image, and return a new image that become
    smaller by 3. to make sure that the new image is similar to the orgin,
    I used the average filter - 1/9 of every value in the filter"""
    small_image = []
    for row_index in range(1, len(image)-1, 3):
        new_line = []
        for col_index in range(1, len(image[0])-1, 3):
            # Now, every image[row_index][col_index] pixel will be in the
            # center of every 3*3 matrix in the image.
            value = int(middle(row_index, col_index, AVG_FILTER, image))
            new_line.append(value)
        small_image.append(new_line)

    return small_image


def downsample(image, max_diagonal_size):
    """Function that gets 3 parameters - image and diagonal size, and return
    a new image that become smaller - depend the diagonal input. its uses
    the downsample_by_3 function"""
    small_image = copy.deepcopy(image)
    width = len(image[0])
    height = len(image)
    diagonal = (width**2 + height**2)**(1/2)
    while max_diagonal_size <= diagonal:
        small_image = downsample_by_3(small_image)
        new_width = len(small_image[0])
        new_height = len(small_image)
        diagonal = (new_width**2 + new_height**2)**(1/2)

    return small_image


def values_of_pixels(image, list_index):
    """Function that gets image and list of index (the list is the output of
    the pixels_on_line from aligned_helper.py), and return the sum of all
    (len)**2, len is the length of all the white pixels that enough close each
    other"""
    sum = 0
    last_index = [0,0]
    new_list = []
    for location in list_index:
        if image[location[0]][location[1]] == WHITE_INDEX:
            if new_list == []:
                # The first location will be append to the list without
                # Preconditions
                new_list.append(location)
            elif distance(last_index, location) <= MAX_DIST:
                # Here we start to build the sequence
                new_list.append(location)
            else:
                length = distance(new_list[0], new_list[-1])
                sum += length**2
                new_list = [location]  # Strating to build a new list
            last_index = location

    if len(new_list) > 1:
        # if the last index in the loop is white, and its part of list that
        # big than one, the next loop will be out of range, so we need to
        # add this manual.
        length = distance(new_list[0], new_list[-1])
        sum += length ** 2

    return sum


def distance(location1, location2):
    """Function that gets two location (row and column) and return the
    distance between them"""
    return math.sqrt((location1[0]-location2[0])**2 + (location1[1]-location2[
        1])**2)


def get_angle(image):
    """This function get an image and return the dominant angle of the image"""
    width = len(image[0])
    height = len(image)
    diagonal = (width**2 + height**2)**(1/2)
    max = 0
    angle_to_return = 0

    # In this loop we check every angle from 0 degree to 179 degree
    for angle in range(180):
        sum = 0
        angle_in_radian = math.radians(angle)

        # Here we running on all the legal distances
        for dist in range(int(diagonal)+1):

            # If the angle is in this range, we need to take two times the
            # list of the same distance, one time with True and one time
            # with False
            if 0 < angle < 90:
                if dist == 0:
                    lst0 = aligned_helper.pixels_on_line(image, angle_in_radian, 0)
                    sum += values_of_pixels(image, lst0)
                else:
                    lst1 = aligned_helper.pixels_on_line(image, angle_in_radian,
                                                         dist)
                    lst2 = aligned_helper.pixels_on_line(image, angle_in_radian,
                                                         dist, False)
                    sum += values_of_pixels(image, lst2) + values_of_pixels(
                        image, lst1)

            else:
                lst3 = aligned_helper.pixels_on_line(image, angle_in_radian, dist)
                sum += values_of_pixels(image, lst3)

        # Every loop we check if sum is bigger then max, if it is, we put
        # sum value in max. then we put the angle of this distance,
        # to be the angle that we want to return
        if max < sum:
            max = sum
            angle_to_return = angle

    return angle_to_return


def moving(row, col, angle):
    """This is a help function that gets three parameters: row, col,
    and angle, and return the new place of point from the beginning, depend
    the angle"""
    x = int((math.cos(math.radians(angle))) * row + (math.sin(math.radians(
        angle))) * col)
    y = int((-math.sin(math.radians(angle))) * row + (math.cos(math.radians(
        angle))) * col)
    return x, y


def rotate(image, angle):
    """Function that gets two parameters: image and angle, and return a new
    image after rotation by the angle"""
    new_image = []
    # I build this function for rotate against the clock, than I saw that
    # its should be with the clock, so I put (-1) in every use of 'angle'.
    angle_in_rad = math.radians(-angle)
    original_width = len(image[0])
    original_height = len(image)

    # We must build a new image in the correct size, and the height and the
    # width has to be integer, and the angles must be in absolute value
    new_height = int((abs(math.cos(abs(angle_in_rad))))*original_height + (abs(
        math.sin(abs(angle_in_rad)))*original_width))
    new_width = int((abs(math.cos(abs(angle_in_rad)))*original_width +
                     (abs(math.sin(abs(angle_in_rad))))*original_height))

    # Building a new image in the new size (after rotation we get new size)
    for i in range(new_height):
        new_line = []
        for j in range(new_width):
            new_line.append(BLACK_INDEX)  # All the pixels will be blacks
        new_image.append(new_line)

    for row in range(len(new_image)):
        for col in range(len(new_image[0])):
            # Using moving function after we moce (row,col) to the center of
            #  the new image that we build:
            x, y = moving(row - (new_height // 2), col - (new_width // 2),
                          -angle)
            # Returning to the correct place in original image:
            x += original_height//2
            y += original_width//2

            if 0 <= x < original_height and 0 <= y < original_width:
                new_image[row][col] = image[x][y]

    return new_image


def make_correction(image, max_diagonal):
    """Function that gets two parameters: image and max diagonal size,
    and return a new aligned image"""
    new_image = copy.deepcopy(image)
    image = downsample(image, max_diagonal)
    image = threshold_filter(image)
    image = detect_edges(image)
    image = threshold_filter(image)
    dominant_angle = get_angle(image)
    return rotate(new_image, -dominant_angle)


def main():
    if len(sys.argv) != NUM_OF_PARAM + 1:
        print(ERROR_MESSAGE)
        sys.exit()

    image_source = sys.argv[1]
    output_name = sys.argv[2]
    max_diagonal = int(sys.argv[3])
    image = aligned_helper.load_image(image_source)
    aligned_helper.save(make_correction(image, max_diagonal), output_name)


if __name__ == "__main__":
    main()





