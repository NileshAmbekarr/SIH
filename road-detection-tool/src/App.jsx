import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './Components/Navbar';
import Login from './Pages/Login';
import SelectAOI from './Pages/SelectAOI';
import ViewAnalytics from './Pages/ViewAnalytics';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/select-aoi" element={<SelectAOI />} />
        <Route path="/analytics" element={<ViewAnalytics />} />
      </Routes>
    </Router>
  );
}

export default App;
