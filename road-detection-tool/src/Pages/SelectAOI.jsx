import React, { useState } from 'react';
import Dropdown from '../Components/Dropdown';
import AOISelectionMap from '../Components/AOISelectionMap';
import ShapefileDisplay from '../Components/ShapeFileDisplay';

const SelectAOI = () => {
  const [state, setState] = useState('');
  const [district, setDistrict] = useState('');
  const [taluka, setTaluka] = useState('');
  const [selectedAOI, setSelectedAOI] = useState(null);

  const handleAOISelection = (aoi) => {
    setSelectedAOI(aoi);
  };

  return (
    <div>
      <h1>Select Area of Interest (AOI)</h1>
      <Dropdown label="State" options={['Maharashtra', 'State 2']} onSelect={setState} />
      <Dropdown label="District" options={['District 1', 'District 2']} onSelect={setDistrict} />
      <Dropdown label="Taluka" options={['Taluka 1', 'Taluka 2']} onSelect={setTaluka} />
      
      <AOISelectionMap onAOISelect={handleAOISelection} />
      {selectedAOI && <ShapefileDisplay aoi={selectedAOI} />}
    </div>
  );
};

export default SelectAOI;

