# The sys module provides access to some variables used or maintained by the Python interpreter and functions that interact strongly with the interpreter,  It is often used to manipulate the Python runtime environment, access command-line arguments, or modify the behavior of the Python interpreter
import sys
# The os module provides a way of interacting with the operating system. It allows you to perform various operating system-related tasks, such as reading or changing the current working directory, interacting with the file system, and running system commands, file and directory operations, environment variables, etc. 
import os
# The json module is used for encoding and decoding JSON (JavaScript Object Notation) data, It is used to Reading JSON data from a file, converting Python objects to JSON format, and vice versa.
import json
# The xml.etree.ElementTree module is part of the standard library and provides a simple and lightweight XML (eXtensible Markup Language) parsing library, It is use in parsing and manipulating XML data
import xml.etree.ElementTree as ET
# The glob module provides a function for finding all the pathnames matching a specified pattern according to the rules used by the Unix shell, It is used for getting a list of files that match a certain pattern.
import glob

# This constant is set to 1 and represents the starting value for the bounding box IDs. Bounding box IDs are used to uniquely identify each bounding box in the COCO format.
START_BOUNDING_BOX_ID = 1
# Initially, the script sets PRE_DEFINE_CATEGORIES to None, indicating that there are no pre-defined categories. This means that the script will dynamically generate categories based on the annotations present in the provided Pascal VOC XML files.
PRE_DEFINE_CATEGORIES = None

# This function takes in the the root variable is an object. It represents the root element of an XML tree and then the string representation of the tag or the elemetnts you are looking for within the given root element
def get(root, name):
    # The findall method is used to find all elements in the XML tree with a specific tag name (name), vars is going to be a list containg the memory adress of each element
    vars = root.findall(name)
    return vars

# This function gets the elements that have the name "name" and also checks if they are up to a certain number "length"
def get_and_check(root, name, length):
    # Use findall to find all elements with the specified tag name
    vars = root.findall(name)

    # Check if no elements were found, raise an error if so
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))

    # Check if the length of found elements does not match the expected length
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )

    # If the expected length is 1 and only one element is found, assign it to vars
    if length == 1:
        vars = vars[0]

    # Return the list of found elements or the single element if length was 1
    return vars

# This function processes a filename and converts it to a string, removing its extension and replacing backslashes with forward slashes
def get_filename_as_int(filename):
    try:
        # replaces "//" with "/"
        filename = filename.replace("\\", "/")
        # Splits the text based on the file name, it uses os.path.basename to get the base name of the file (without the extension) and os.path.splitext to split the filename into its root and extension. [0] is then used to get the root part. This is done to remove the file extension.
        filename = os.path.splitext(os.path.basename(filename))[0]
        # Returns only the file name as a string
        return str(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))
    
def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    # Initialize an empty list to store category names
    classes_names = []

    # Iterate over each XML file in the list
    for xml_file in xml_files:
        # Parse the XML file and get the root element
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Iterate over each "object" element in the XML tree
        for member in root.findall("object"):
            # Append the text content of the first child element to the list
            classes_names.append(member[0].text)

    # Remove duplicate category names, sort the list, and create a dictionary with name to id mapping
    classes_names = list(set(classes_names))
    classes_names.sort()
    
    # Create a dictionary with category name to id mapping using enumerate
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    # Initialize the JSON dictionary with placeholders for images, annotations, and categories
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    # Check if predefined categories are provided, otherwise generate them from XML files
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)

    # Initialize bounding box ID and iterate over each XML file
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        # Parse the XML file and get the root element
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract the path element from the XML tree
        path = get(root, "path")

        # Determine the filename based on the path or use the "filename" element
        # Making sure it's just one path element that is in that file
        if len(path) == 1:
            # "os.path.basename()" extracts the base filename from the given path. It removes the directory part and returns just the filename, "path[0].text" accesses the text content of the first path element in the XML tree. In XML annotation files, the path element typically specifies the file path or location of the image.
            filename = os.path.basename(path[0].text)
        # It checks if there are not path element in the file
        elif len(path) == 0:
            # If there are no path element it checks runs the get_check_function and will throw error
            filename = get_and_check(root, "filename", 1).text
        # Raises error if there are more than 1 path element
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        # The purpose of this line is to generate a unique identifier (image_id) based on the filename, which will be used in the subsequent parts of the code, such as creating annotations for the image.
        image_id = get_filename_as_int(filename)

        # Extract image size information using the get_check function
        # Here we are making sure that the size element is just 1 as it is going to be the root for element to get width and height
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        # Create image dictionary and append it to the JSON images list
        image = {"file_name": filename, "height": height, "width": width, "id": image_id}
        json_dict["images"].append(image)

        # Iterate over "object" elements in the XML tree (representing annotations)
        for obj in get(root, "object"):
            # Extract category name and handle new categories
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]

            # Extract bounding box coordinates
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)

            # Ensure valid bounding box coordinates
            assert xmax > xmin
            assert ymax > ymin

            # Calculate bounding box dimensions
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            # Create annotation dictionary and append it to the JSON annotations list
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)

            # Increment bounding box ID
            bnd_id = bnd_id + 1

    # Create category dictionaries and append them to the JSON categories list
    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    # Write the JSON dictionary to a file
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == "__main__":
    # Import necessary modules
    import argparse

    # Create an argument parser with a description
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC annotation to COCO format."
    )
    
    # Define command-line arguments
    parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Get a list of XML files in the specified directory
    xml_files = glob.glob(os.path.join(args.xml_dir, "*.xml"))

    # Print the number of XML files found
    print("Number of xml files: {}".format(len(xml_files)))
    
    # Convert Pascal VOC annotation to COCO format using the convert function
    convert(xml_files, args.json_file)
    
    # Print a success message with the output json file name
    print("Success: {}".format(args.json_file))