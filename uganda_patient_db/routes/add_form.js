const express = require('express');
const router = express.Router();
const form_template = require('../services/form_template');




/* POST quotes */
router.post('/', async function(req, res, next) {
  try {
        console.log(`req body`, req.body)
        res.json(await form_template.create(req.body));
  } catch (err) {
    console.error(`Error while posting a new patient `, err.message);
    next(err);
  }
});




module.exports = router;