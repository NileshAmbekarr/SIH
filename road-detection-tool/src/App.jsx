import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Navbar from './Components/Navbar';
import Login from './Pages/Login';
import SelectAOI from './Pages/SelectAOI';
import ViewAnalytics from './Pages/ViewAnalytics';
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(true);

  useEffect(() => {
    // Check for auth token or session on component load
    const token = localStorage.getItem('authToken');
    if (token) {
      setIsAuthenticated(false);
    }
  }, []);

  return (
    <div>
      {isAuthenticated ? (
        <Router>
          <Navbar />
          <Routes>
            <Route path="/select-aoi" element={<SelectAOI />} />
            <Route path="/analytics" element={<ViewAnalytics />} />
          </Routes>
        </Router>
      ) : (
        <Login setIsAuthenticated={setIsAuthenticated} />
      )}
    </div>
  );
}


export default App;

