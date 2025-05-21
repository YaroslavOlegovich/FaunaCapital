require('dotenv').config();
const express = require('express');
const path = require('path');
const session = require('express-session');
const passport = require('passport');
const bcrypt = require('bcryptjs');
const sqlite3 = require('sqlite3').verbose();
const LocalStrategy = require('passport-local').Strategy;

const app = express();

// Подключение к SQLite
const db = new sqlite3.Database('./fauna.db', (err) => {
  if (err) return console.error('Ошибка подключения к SQLite:', err.message);
  console.log('База данных SQLite подключена.');
});

// Создание таблицы пользователей
db.run(`
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Настройка шаблонов
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'html');
app.engine('html', require('ejs').renderFile);

// Статика и парсинг форм
app.use(express.static(path.join(__dirname, 'views')));
app.use(express.urlencoded({ extended: true }));

// Сессии
app.use(session({
  secret: process.env.SESSION_SECRET || 'secret',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    maxAge: 24 * 60 * 60 * 1000 // 24 часа
  }
}));

app.use(passport.initialize());
app.use(passport.session());

// Passport стратегия
passport.use(new LocalStrategy({ usernameField: 'email' }, (email, password, done) => {
  db.get('SELECT * FROM users WHERE email = ?', [email], async (err, user) => {
    if (err) return done(err);
    if (!user) return done(null, false, { message: 'Пользователь не найден' });
    
    try {
      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) return done(null, false, { message: 'Неверный пароль' });
      return done(null, user);
    } catch (err) {
      return done(err);
    }
  });
}));

passport.serializeUser((user, done) => done(null, user.id));
passport.deserializeUser((id, done) => {
  db.get('SELECT * FROM users WHERE id = ?', [id], (err, user) => {
    done(err, user);
  });
});

// Middleware для проверки аутентификации
const isAuthenticated = (req, res, next) => {
  if (req.isAuthenticated()) return next();
  res.redirect('/login');
};

// Роуты
app.get('/', (req, res) => {
  res.render('index');
});

app.get('/register', (req, res) => {
  res.render('register');
});

app.post('/register', async (req, res) => {
  const { name, email, password } = req.body;
  
  try {
    // Проверка существования пользователя
    db.get('SELECT * FROM users WHERE email = ?', [email], async (err, user) => {
      if (err) {
        console.error(err);
        return res.status(500).send('Ошибка сервера');
      }
      
      if (user) {
        return res.status(400).send('Пользователь с таким email уже существует');
      }
      
      // Хеширование пароля
      const hashedPassword = await bcrypt.hash(password, 10);
      
      // Создание пользователя
      db.run(
        'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
        [name, email, hashedPassword],
        function(err) {
          if (err) {
            console.error(err);
            return res.status(500).send('Ошибка при регистрации');
          }
          
          // Автоматический вход после регистрации
          db.get('SELECT * FROM users WHERE id = ?', [this.lastID], (err, user) => {
            if (err) return res.status(500).send('Ошибка сервера');
            req.login(user, (err) => {
              if (err) return res.status(500).send('Ошибка сервера');
              res.redirect('/dashboard');
            });
          });
        }
      );
    });
  } catch (err) {
    console.error(err);
    res.status(500).send('Ошибка сервера');
  }
});

app.get('/login', (req, res) => {
  res.render('login');
});

app.post('/login', passport.authenticate('local', {
  successRedirect: '/dashboard',
  failureRedirect: '/login',
  failureFlash: true
}));

app.get('/dashboard', isAuthenticated, (req, res) => {
  res.render('dashboard', { user: req.user });
});

app.get('/logout', (req, res) => {
  req.logout((err) => {
    if (err) return next(err);
    res.redirect('/');
  });
});

// Запуск сервера
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Сервер запущен на http://localhost:${PORT}`);
});