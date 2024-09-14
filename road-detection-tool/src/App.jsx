import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { useState ,useEffect } from 'react';
import Navbar from './Components/Navbar';
import Login from './Pages/Login';
import SelectAOI from './Pages/SelectAOI';
import ViewAnalytics from './Pages/ViewAnalytics';
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Check for auth token or session on component load
    const token = localStorage.getItem('authToken');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  return (
    <div>
      {isAuthenticated ? (
        <>
          <selectAOI />
          <viewAnalytics />
        </>
      ) : (
        <Login setIsAuthenticated={setIsAuthenticated} />
      )}
    </div>
  );
}

export default App;

