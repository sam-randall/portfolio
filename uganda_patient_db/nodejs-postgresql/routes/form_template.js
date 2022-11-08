const express = require('express');
const router = express.Router();
const form_template = require('../services/form_template');

/* POST form fields and name */

// res :response - we fill that out.
// req: client-provided in Terminal or otherwise, data
// --data '{"field1":"doggy"}'
router.post('/create/', async function(req, res, next) {
  try {
        console.log(`Received data from POST`, req.body)
        res.json(await form_template.create(req.body));
  } catch (err) {
    console.error(`Error while posting a new patient `, err.message);
    next(err);
  }
});


router.post('/add/', async function(req, res, next) {
  try {
        console.log(`Received data from POST`, req.body)
        let body = req.body
        res.json(await form_template.add_form_for_patient(body.nin, body.name, body.fields));
  } catch (err) {
    console.error(`Error while posting a new patient `, err.message);
    next(err);
  }
});




module.exports = router;