import { useState } from "react";

// Sample data for states, districts, and talukas
const locationData = {
  Maharashtra: {
    Pune: ["Taluka1", "Taluka2"],
    Mumbai: ["Taluka3", "Taluka4"],
  },
  Gujarat: {
    Ahmedabad: ["Taluka5", "Taluka6"],
    Surat: ["Taluka7", "Taluka8"],
  },
};

function SelectAOI() {
  const [state, setState] = useState("");
  const [district, setDistrict] = useState("");
  const [taluka, setTaluka] = useState("");

  const handleStateChange = (e) => {
    setState(e.target.value);
    setDistrict(""); // Reset district when state changes
    setTaluka(""); // Reset taluka when state changes
  };

  const handleDistrictChange = (e) => {
    setDistrict(e.target.value);
    setTaluka(""); // Reset taluka when district changes
  };

  return (
    <div>
      <label>State:</label>
      <select value={state} onChange={handleStateChange}>
        <option value="">Select State</option>
        {Object.keys(locationData).map((state) => (
          <option key={state} value={state}>
            {state}
          </option>
        ))}
      </select>

      {state && (
        <>
          <label>District:</label>
          <select value={district} onChange={handleDistrictChange}>
            <option value="">Select District</option>
            {Object.keys(locationData[state]).map((district) => (
              <option key={district} value={district}>
                {district}
              </option>
            ))}
          </select>
        </>
      )}

      {district && (
        <>
          <label>Taluka:</label>
          <select value={taluka} onChange={(e) => setTaluka(e.target.value)}>
            <option value="">Select Taluka</option>
            {locationData[state][district].map((taluka) => (
              <option key={taluka} value={taluka}>
                {taluka}
              </option>
            ))}
          </select>
        </>
      )}
    </div>
  );
}

export default SelectAOI;

