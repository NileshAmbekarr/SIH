import React from 'react';
import './ShapeFileDisplay.css'

const ShapefileDisplay = ({ aoi }) => {
  // Placeholder for shapefile visualization logic
  return (
    <div className='shapefile-container'>
      <h2>Shapefile Data for Selected AOI</h2>
      <p className='shapefile-item'>Displaying roads for selected area with bounds: {aoi.toBBoxString()}</p>
      {/* Add shapefile display logic here */}
    </div>
  );
};

export default ShapefileDisplay;
