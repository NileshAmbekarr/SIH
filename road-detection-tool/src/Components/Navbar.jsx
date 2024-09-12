import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css'
const Navbar = () => {
  return (
    <nav  className='wrapper'>
      <ul className='list-wrapper'>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/analytics">View Analytics</Link></li>
        <li><Link to="/select-aoi">Select AOI</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
