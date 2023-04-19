import React, { useEffect } from 'react'
import { Container, Row, Col, Button } from 'react-bootstrap'
import { useState } from 'react';
import './App.css';


function App() {
  const [price, setPrice] = useState('')

  const predictPrice = () => {
    // Make an AJAX request to backend Python function
    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      // body: JSON.stringify({
      //     param1: 'value1',
      //     param2: 'value2'
      // }),
      headers: {
          'Content-Type': 'application/json'
      }
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        setPrice(data["result"]);
    }).catch(error => {
      console.log(error)
    });
  }

  useEffect(() => {
    
  }, [price]);

  return (
    <Container>
      <Row>
        <Col sm={5}>
          <Button onClick={predictPrice}>Predict tomorrow's price</Button>
          <h3>Predicted price: {price}</h3>
        </Col>
        <Col sm={5}>
          hi
        </Col>
      </Row>
    </Container>
  );
}

export default App;
