import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/analytics">View Analytics</Link></li>
        <li><Link to="/select-aoi">Select AOI</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;
