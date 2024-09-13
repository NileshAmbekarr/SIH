import React from 'react';
import './Dropdown.css'

const Dropdown = ({ label, options, onSelect }) => {
  return (
    <div className='dropdown'>
      <label>{label}:</label>
      <select onChange={(e) => onSelect(e.target.value)}>
        <option value="">Select {label}</option>
        {options.map((option, index) => (
          <option key={index} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
};

export default Dropdown;
