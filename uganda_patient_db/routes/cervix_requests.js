const express = require('express');
const router = express.Router();
const cervix_requests = require('../services/cervix_requests');

/* GET patients listing. */
router.get('/', async function(req, res, next) {
  try {
    res.json(await cervix_requests.getRequestHistoryForPatient(req.query.id));
  } catch (err) {
    console.error(`Error while getting patients `, err.message);
    next(err);
  }
});


/* POST quotes */
router.post('/', async function(req, res, next) {
  try {
    res.json(await cervix_requests.createRequest(req.body));
  } catch (err) {
    console.error(`Error while posting a new patient `, err.message);
    next(err);
  }
});

module.exports = router;