import xml.etree.ElementTree as ET
import pandas as pd

# Load the SVG file
tree = ET.parse(r"C:\Users\Windows\Desktop\output.svg")

# Load the SVG file
root = tree.getroot()

# Initialize lists to store the data
cx_values = []
cy_values = []
r_values = []

# Access elements in the SVG
for element in root:
    # Check if the element is a circle
    if element.tag == '{http://www.w3.org/2000/svg}circle':
        cx = float(element.get('cx'))
        cy = float(element.get('cy'))
        r = float(element.get('r'))
        
        # Append the values to the lists
        cx_values.append(cx)
        cy_values.append(cy)
        r_values.append(r)

# Create a DataFrame
data = {
    'cx': cx_values,
    'cy': cy_values,
    'r': r_values
}
df = pd.DataFrame(data)


# Display the DataFrame
print(df)
plt.plot(df.cx, df.cy,'.')
df.to_csv(r'Y:\2020-09-Paper-ReactiveTouch\_randomPDF_and_Output\outwhiskall.csv')



# Create a new SVG document
svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", xmlns:xlink="http://www.w3.org/1999/xlink", version="1.1", width="100%", height="100%", viewBox="0 0 452.5 255.6")

# Define the style for the circles
style = ET.SubElement(svg, 'style')
style.set('type', 'text/css')
style.text = '.st0{fill:none;stroke:#000000;stroke-width:0.4804;stroke-miterlimit:10;}'

# Iterate through the DataFrame and create circles
for index, row in df.iterrows():
    circle = ET.SubElement(svg, 'circle')
    circle.set('cx', str(row['cx']))
    circle.set('cy', str(row['cy']))
    circle.set('r', str(row['r']))

# Create an ElementTree object and save it to a file
tree = ET.ElementTree(svg)
tree.write('output.svg', encoding='utf-8', xml_declaration=True)