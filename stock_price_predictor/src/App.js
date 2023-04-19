import React, { useEffect } from 'react'
import { Container, Row, Col, Button, Form } from 'react-bootstrap'
import { useState } from 'react';
import './App.css';
import Loader from './Components/Loader';


function App() {
  const [price, setPrice] = useState('')
  const [ytdPrice, setYtdPrice] = useState('')

  const predictPrice = (e) => {
    e.preventDefault()
    setPrice(null)
    // Make an AJAX request to backend Python function
    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: JSON.stringify({
          ytdPrice: ytdPrice,
      }),
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
    
  }, [price, ytdPrice]);

  return (
    <Container className='App'>
      <Row>
        <Col sm={5}>
          <Form onSubmit={predictPrice}>
            <Form.Group controlId="ytdPrice">
              <Form.Label>Yesterday's Closing Price: </Form.Label> {price == null && <Loader />}
              <Form.Control
                type="name"
                placeholder="Enter yesterday's price"
                value={ytdPrice}
                onChange={(e) => setYtdPrice(e.target.value)}
              ></Form.Control>
            </Form.Group>

            <Button type="submit" variant="light">
              Predict today's stock price
            </Button>

          </Form>
          <h3>Predicted price: {price}</h3>
        </Col>
        <Col sm={5}>
          
        </Col>
      </Row>
    </Container>
  );
}

export default App;
