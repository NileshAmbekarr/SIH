import React, { useRef, useEffect } from 'react';
import L from 'leaflet';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';

const AOISelectionMap = ({ onAOISelect }) => {
  const mapRef = useRef(null);
  let mapInstance = useRef(null); // Store the map instance

  useEffect(() => {
    if (!mapInstance.current) {
      mapInstance.current = L.map(mapRef.current).setView([51.505, -0.09], 13); // Create the map

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
      }).addTo(mapInstance.current);

      const drawnItems = new L.FeatureGroup();
      mapInstance.current.addLayer(drawnItems);

      const drawControl = new L.Control.Draw({
        draw: {
          polygon: false,
          polyline: false,
          circle: false,
          marker: false,
          rectangle: true,
        },
        edit: {
          featureGroup: drawnItems,
        },
      });

      mapInstance.current.addControl(drawControl);

      mapInstance.current.on('draw:created', (e) => {
        const layer = e.layer;
        drawnItems.addLayer(layer);
        const bounds = layer.getBounds();
        onAOISelect(bounds);
      });
    }

    return () => {
      // Clean up map instance
      if (mapInstance.current) {
        mapInstance.current.off();
        mapInstance.current.remove();
      }
    };
  }, [onAOISelect]);

  return <div id="map" ref={mapRef} style={{ height: '500px', width: '100%' }} />;
};

export default AOISelectionMap;
