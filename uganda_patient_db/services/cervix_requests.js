const db = require('./db');
const helper = require('../helper');


// POST - add cervix request to db
async function createRequest(request) {
    // validateCreate(patient)
    const result = await db.query(
      'INSERT INTO cervix_requests (nin, date, via_result, needs_biopsy, map, notes, next_review_date, screening_staff_name) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING *',
      [request.nin, request.date, request.via_result, request.needs_biopsy, request.map, request.notes, request.next_review_date, request.screening_staff_name]
    );
    let message = 'Error in creating quote';
  
    if (result.length) {
      message = 'Cervix request created successfully';
    }
  
    return {message};
}

// GET - get all for one.
async function getRequestHistoryForPatient(id) {
    const rows = await db.query(
        'SELECT * FROM cervix_requests WHERE nin = $1', 
        [id]
      );

    const data = helper.emptyOrRows(rows);
    const meta = {id};

    return {
        data,
        meta
    }
}

module.exports = {
    getRequestHistoryForPatient,
    createRequest
  }

