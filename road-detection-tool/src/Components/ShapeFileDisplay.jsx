import React from 'react';

const ShapefileDisplay = ({ aoi }) => {
  // Placeholder for shapefile visualization logic
  return (
    <div>
      <h2>Shapefile Data for Selected AOI</h2>
      <p>Displaying roads for selected area with bounds: {aoi.toBBoxString()}</p>
      {/* Add shapefile display logic here */}
    </div>
  );
};

export default ShapefileDisplay;
