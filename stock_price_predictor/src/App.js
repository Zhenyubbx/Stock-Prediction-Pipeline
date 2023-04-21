import React, { useEffect } from 'react'
import { Container, Row, Col, Button, Form, Dropdown } from 'react-bootstrap'
import { useState } from 'react';
import './App.css';
import Loader from './Components/Loader';


function App() {
  const [price, setPrice] = useState('')
  const [tdyPrice, setYtdPrice] = useState('')
  const [stock, setStock] = useState('')

  const predictPrice = (e) => {
    e.preventDefault()
    setPrice(null)
    // Make an AJAX request to backend Python function
    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: JSON.stringify({
          tdyPrice: tdyPrice,
          stock: stock,
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
    
  }, [price, tdyPrice, stock]);

  return (
    <Container className='App'>
      <Row>
        <Col sm={5}>
        {price!=null ? (
          <Col>
          <Form onSubmit={predictPrice}>
            
            <Form.Group controlId="stock">
              <Form.Label>Select a stock: </Form.Label>
              <Form.Select onChange={(e) => setStock(e.target.value)}>
                <option value="">Choose...</option>
                <option value="AAPL">AAPL</option>
                <option value="AMD">AMD</option>
                <option value="AMZN">AMZN</option>
                <option value="BA">BA</option>
                <option value="BX">BX</option>
                <option value="COST">COST</option>
                <option value="CRM">CRM</option>
                <option value="DIS">DIS</option>
                <option value="ENPH">ENPH</option>
                <option value="F">F</option>
                <option value="GOOG">GOOG</option>
                <option value="INTC">INTC</option>
                <option value="KO">KO</option>
                <option value="META">META</option>
                <option value="MSFT">MSFT</option>
                <option value="NFLX">NFLX</option>
                <option value="NIO">NIO</option>
                <option value="NOC">NOC</option>
                <option value="PG">PG</option>
                <option value="PYPL">PYPL</option>
                <option value="TSLA">TSLA</option>
                <option value="TSM">TSM</option>
                <option value="VZ">VZ</option>
                <option value="XPEV">XPEV</option>
                <option value="ZS">ZS</option>
              </Form.Select>
            </Form.Group>

            <Form.Group controlId="tdyPrice">
              <Form.Label>Today's Closing Price: </Form.Label>
              <Form.Control
                type="name"
                placeholder="Enter today's price"
                value={tdyPrice}
                onChange={(e) => setYtdPrice(e.target.value)}
              ></Form.Control>
            </Form.Group>

            <Button type="submit" variant="light">
              Predict tomorrow's stock price
            </Button>

          </Form>
          <h3>Predicted price: {price!=null && price!='' && '$'}{price}</h3>
          </Col>
        ) : <Loader /> }
          
        </Col>
        <Col sm={5}>
          
        </Col>
      </Row>
    </Container>
  );
}

export default App;
