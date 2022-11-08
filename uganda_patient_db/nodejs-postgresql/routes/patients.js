const express = require('express');
const router = express.Router();
const patients = require('../services/patients');

/* GET patients listing. */
router.get('/', async function(req, res, next) {
  try {
    res.json(await patients.getOneRecord(req.query.id));
  } catch (err) {
    console.error(`Error while getting patients `, err.message);
    next(err);
  }
});

/* GET patient form data for given nin and  */
router.get('/data', async function(req, res, next) {
  try {
    console.log(req.params)
    console.log(req.body)
    console.log(req.query)
    const { id, name } = req.query
    if (id && name) {
      res.json(await patients.getRecordFromNinAndFormName(id, name));
    } else {
      console.log(id, name)
    }
  } catch (err) {
    console.error(`Error while getting patients `, err.message);
    next(err);
  }
});


/* POST quotes */
router.post('/', async function(req, res, next) {
  try {
    res.json(await patients.create(req.body));
  } catch (err) {
    console.error(`Error while posting a new patient `, err.message);
    next(err);
  }
});




module.exports = router;