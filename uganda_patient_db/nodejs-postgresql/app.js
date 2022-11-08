var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var patientsRouter = require('./routes/patients');
var cervixRequestsRouter = require('./routes/cervix_requests')
var formRouter = require('./routes/form_template')

var app = express();

process.title = "myApp"

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/patients', patientsRouter);
app.use('/form', formRouter)

app.listen(3000, '0.0.0.0');
module.exports = app;
